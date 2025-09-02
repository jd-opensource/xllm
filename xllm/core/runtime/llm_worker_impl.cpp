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

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>

#include "pytorch/adapter/utils/utils.h"
#endif

#include <memory>
#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool LLMWorkerImpl::init_model(torch::ScalarType dtype,
                               const ModelArgs& model_args,
                               const QuantArgs& quant_args) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
#if defined(USE_NPU)
  int currentDevId = device_.index();
  int ret = aclrtSetDevice(currentDevId);
  if (ret != 0) {
    LOG(ERROR) << "ACL set device id:" << currentDevId
               << " failed, ret:" << ret;
  }
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu init device
#endif

  // initialize model
  context_.set_model_args(model_args);
  context_.set_quant_args(quant_args);
  dtype_ = dtype;
  context_.set_tensor_options(torch::dtype(dtype_).device(device_));
  // Try to create a causal LM model
  model_ = create_llm_model(context_);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ =
      std::make_unique<Executor>(model_.get(), model_args, device_, options_);

  eplb_executor_ = std::make_unique<EplbExecutor>(model_.get());
  return true;
}

std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& inputs) {
#if defined(USE_NPU)
  c10_npu::SetDevice(device_.index());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu set device
#endif
  Timer timer;
  auto& flatten_tokens = inputs.token_ids;
  auto& flatten_positions = inputs.positions;
  auto& params = inputs.input_params;
  auto& sampling_params = inputs.sampling_params;

  if (FLAGS_enable_eplb) {
    eplb_executor_->eplb_execute(inputs.eplb_info);
  }
  std::vector<folly::SemiFuture<bool>> futures;
  if (options_.instance_role() == InstanceRole::PREFILL &&
      options_.kv_cache_transfer_mode() == "PUSH" &&
      !inputs.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
    const_cast<ModelInputParams*>(&params)->layer_synchronizer =
        layer_synchronizer;

    futures.emplace_back(
        kv_cache_transfer_->push_kv_blocks_async(inputs.transfer_kv_infos,
                                                 context_.get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_));
#endif
  }

  // call model executor forward to get hidden states
  auto hidden_states = model_executor_->forward(
      flatten_tokens, flatten_positions, kv_caches_, params);
  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits =
        model_->logits(hidden_states, sampling_params.selected_token_idxes);
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
#if defined(USE_NPU)
    // torch::npu::synchronize();
    aclrtSynchronizeStream(
        c10_npu::getCurrentNPUStream(device_.index()).stream());
#elif defined(USE_MLU)
    // TODO(mlu): implement mlu synchronize stream
#endif
    if (options_.instance_role() == InstanceRole::PREFILL &&
        options_.kv_cache_transfer_mode() == "PUSH" &&
        !inputs.transfer_kv_infos.empty()) {
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
  if (sampling_params.selected_token_idxes.defined()) {
    sample_output = sampler_->forward(logits, sampling_params);
    output.logits = logits;

    if (options_.enable_speculative_decode()) {
      if (params.empty_kv_cache) {
        sample_output.embeddings = hidden_states;
      } else {
        auto sample_idxes = sampling_params.selected_token_idxes.index_select(
            /*dim=*/0, sampling_params.sample_idxes);
        auto embeddings = hidden_states.index_select(/*dim=*/0, sample_idxes);
        sample_output.embeddings = embeddings;
      }
    }

    // set sample output to output
    output.sample_output = sample_output;
    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

#if defined(USE_NPU)
  aclrtSynchronizeStream(
      c10_npu::getCurrentNPUStream(device_.index()).stream());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu synchronize stream
#endif

  if (options_.instance_role() == InstanceRole::PREFILL &&
      options_.kv_cache_transfer_mode() == "PUSH" &&
      !inputs.transfer_kv_infos.empty()) {
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
