/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA)
#include "layers/common/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA)
  // initialize flashinfer workspace
  layer::FlashinferWorkspace::get_instance().initialize(device_);
#endif
}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // Try to create a causal LM model
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

std::optional<ForwardOutput> LLMWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
  Timer timer;
  std::vector<torch::Tensor> flatten_tokens_micro_batches;
  std::vector<torch::Tensor> flatten_positions_micro_batches;
  std::vector<ModelInputParams> input_params_micro_batches;
  auto& concated_sampling_params = inputs.concated_sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  for (auto i = 0; i < inputs.micro_inputs.size(); ++i) {
    flatten_tokens_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].token_ids));
    flatten_positions_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].positions));
    input_params_micro_batches.push_back(
        std::move(inputs.micro_inputs[i].input_params));

    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[i].transfer_kv_infos.empty()) {
#if defined(USE_NPU)
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
          std::make_shared<NPULayerSynchronizerImpl>(
              context_.get_model_args().n_layers());
      const_cast<ModelInputParams*>(&(input_params_micro_batches[i]))
          ->layer_synchronizer = layer_synchronizer;

      futures.emplace_back(kv_cache_transfer_->push_kv_blocks_async(
          inputs.micro_inputs[i].transfer_kv_infos,
          context_.get_parallel_args(),
          layer_synchronizer,
          is_spec_draft_));
#endif
    }
  }
  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(inputs.micro_inputs[0].eplb_info);
  }

  // temporarily use [0], will be adapted in next pr
  // call model executor forward to get hidden states
  auto hidden_states = model_executor_->forward(flatten_tokens_micro_batches,
                                                flatten_positions_micro_batches,
                                                kv_caches_,
                                                input_params_micro_batches);
  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (concated_sampling_params.selected_token_idxes.defined()) {
    logits = model_->logits(hidden_states,
                            concated_sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = expert_load_data_;
    output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      eplb_executor_->reset_ready_layer_id();
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    auto ret = device_.synchronize_default_stream();
    // in p-d disaggregation scene, all micro batches should be in same
    // prefill/decode stage, so, to judge transfer_kv_infos.empty,
    // just use micro inputs.micro_inputs[0] here
    if (options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          return std::nullopt;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  SampleOutput sample_output;
  if (concated_sampling_params.selected_token_idxes.defined()) {
    sample_output = sampler_->forward(logits, concated_sampling_params);
    output.logits = logits;

    // beam search kernel
    BeamSearchOutput beam_search_output;
    if (concated_sampling_params.use_beam_search &&
        inputs.acc_logprob.defined() && inputs.acc_logprob.numel() > 0) {
      beam_search_output = beam_searcher_->forward(inputs.acc_logprob,
                                                   sample_output.top_tokens,
                                                   sample_output.top_logprobs);
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = concated_sampling_params.do_sample;
    output.logprobs = concated_sampling_params.logprobs;
    output.max_top_logprobs = concated_sampling_params.max_top_logprobs;
    // set beam search output to output
    output.beam_search_output = beam_search_output;
  }

  // if running in multi_stream_parallel step, all micro batches
  // should be in same prefill stage, so, to judge empty_kv_cache,
  // just use micro batch 0 here
  if (options_.enable_speculative_decode() && !is_spec_draft_) {
    if (input_params_micro_batches[0].q_seq_lens_vec[0] > 1) {
      output.sample_output.embeddings = hidden_states;
    } else if (concated_sampling_params.sample_idxes.defined()) {
      // auto sample_idxes =
      //     concated_sampling_params.selected_token_idxes.index_select(
      //         /*dim=*/0, concated_sampling_params.sample_idxes);
      auto embeddings = hidden_states.index_select(
          /*dim=*/0, concated_sampling_params.sample_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  auto ret = device_.synchronize_default_stream();

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !inputs.micro_inputs[0].transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        return std::nullopt;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      device_.index());

  return output;
}

}  // namespace xllm
