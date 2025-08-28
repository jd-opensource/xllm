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

#include "embed_worker_impl.h"

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>

#include "pytorch/adapter/utils/utils.h"
#endif
#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/timer.h"

namespace xllm {

EmbedWorkerImpl::EmbedWorkerImpl(const ParallelArgs& parallel_args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool EmbedWorkerImpl::init_model(torch::ScalarType dtype,
                                 const ModelArgs& model_args,
                                 const QuantArgs& quant_args) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // initialize model
  context_.set_model_args(model_args);
  context_.set_quant_args(quant_args);
  dtype_ = dtype;
  context_.set_tensor_options(torch::dtype(dtype_).device(device_));
  // Try to create a embedding LM model
  model_ = create_embeddinglm_model(context_);

  // Dont find model in embedding models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ =
      std::make_unique<Executor>(model_.get(), model_args, device_, options_);
  return true;
}

std::optional<ForwardOutput> EmbedWorkerImpl::step(const ForwardInput& inputs) {
  torch::DeviceGuard device_guard(device_);

  Timer timer;

  // all tensors should be on the same device as model
  auto flatten_tokens = inputs.token_ids.to(device_);
  auto flatten_positions = inputs.positions.to(device_);
  auto params = inputs.input_params.to(device_);
  auto sampling_params = inputs.sampling_params.to(device_, dtype_);

  // call model executor forward to get hidden states
  auto hidden_states = model_executor_->forward(
      flatten_tokens, flatten_positions, kv_caches_, params);

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!driver_) {
    return std::nullopt;
  }

  // driver prepare model output
  ForwardOutput output;
  SampleOutput sample_output;
  if (sampling_params.selected_token_idxes.defined() &&
      inputs.sampling_params.is_embeddings) {
    // create embeddings
    timer.reset();
    // cast model_ from Causal model to Embedding model
    EmbeddingLM* em_model = dynamic_cast<EmbeddingLM*>(model_.get());
    auto embeddings =
        em_model->pooler(hidden_states, sampling_params.selected_token_idxes);
    sample_output.embeddings = embeddings;
    COUNTER_ADD(execution_latency_seconds_sampling, timer.elapsed_seconds());

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
  return output;
}

}  // namespace xllm
