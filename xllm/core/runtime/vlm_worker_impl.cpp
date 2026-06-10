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

#include "vlm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

namespace {

void wait_input_ready_events(const ForwardInput& input, const Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait ForwardInput metadata ready event";
}

StreamEventPtr record_current_stream_event(const Device& device) {
  std::unique_ptr<Stream> stream = device.current_stream();
  StreamEventPtr event = stream->record_event();
  if (event == nullptr) {
    stream->synchronize();
  }
  return event;
}

}  // namespace

VLMWorkerImpl::VLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
}

bool VLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // initialize model
  context.set_encoder_embedding_mode(false);
  model_ = create_vlm_model(context);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);
  return true;
}

std::optional<ForwardOutput> VLMWorkerImpl::step(const ForwardInput& input) {
  Timer timer;
  const bool empty_shard =
      input.input_params.meta.num_sequences == 0 &&
      (!input.token_ids.defined() || input.token_ids.numel() == 0);
  if (empty_shard) {
    return ForwardOutput{};
  }

  // TODO guojinrong, to adapt multi stream parallel later
  // call model executor forward to get hidden states
  auto model_output = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  auto& sampling_params = input.sampling_params;
  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = model_->logits(model_output.hidden_states,
                            sampling_params.selected_token_idxes);
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    auto ret = device_.synchronize_default_stream();
    return std::nullopt;
  }

  ForwardOutput output;
  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output = sampler_->forward(logits, sampling_params);
    output.logits = logits;
    COUNTER_ADD(execution_latency_seconds_sampling, timer.elapsed_seconds());

    // set sample output to output
    output.sample_output = sample_output;

    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  if (options_.enable_speculative_decode()) {
    torch::Tensor embeddings;
    if (model_output.aux_hidden_states.defined()) {
      embeddings = model_output.aux_hidden_states;
    } else {
      embeddings = model_output.hidden_states;
    }
    if (!input.input_params.meta.batch_forward_type.is_decode() &&
        !is_spec_draft_) {
      output.sample_output.embeddings = embeddings;
    } else if (sampling_params.selected_token_idxes.defined()) {
      output.sample_output.embeddings = embeddings.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
    }
  }

  auto ret = device_.synchronize_default_stream();
  return output;
}

std::optional<ForwardOutput> VLMWorkerImpl::execute_no_sync_on_stream(
    const ForwardInput& input,
    Stream& compute_stream) {
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  wait_input_ready_events(input, compute_stream);

  Timer timer;
  const bool empty_shard =
      input.input_params.meta.num_sequences == 0 &&
      (!input.token_ids.defined() || input.token_ids.numel() == 0);
  if (empty_shard) {
    return ForwardOutput{};
  }

  auto model_output = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  auto& sampling_params = input.sampling_params;
  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = model_->logits(model_output.hidden_states,
                            sampling_params.selected_token_idxes);
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    return std::nullopt;
  }

  ForwardOutput output;
  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output = sampler_->forward(logits, sampling_params);
    output.logits = logits;
    COUNTER_ADD(execution_latency_seconds_sampling, timer.elapsed_seconds());

    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  if (options_.enable_speculative_decode()) {
    torch::Tensor embeddings;
    if (model_output.aux_hidden_states.defined()) {
      embeddings = model_output.aux_hidden_states;
    } else {
      embeddings = model_output.hidden_states;
    }
    if (!input.input_params.meta.batch_forward_type.is_decode() &&
        !is_spec_draft_) {
      output.sample_output.embeddings = embeddings;
    } else if (sampling_params.selected_token_idxes.defined()) {
      output.sample_output.embeddings = embeddings.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
    }
  }

  output.retained_input = std::make_shared<ForwardInput>(input);
  if (enable_schedule_overlap()) {
    output.ready_event = record_current_stream_event(device_);
  }
  return output;
}

}  // namespace xllm
