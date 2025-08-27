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

#include "embed_vlm_worker_impl.h"

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
#include "options.h"
#include "util/timer.h"

namespace xllm {

EmbedVLMWorkerImpl::EmbedVLMWorkerImpl(const ParallelArgs& parallel_args,
                                       const torch::Device& device,
                                       const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool EmbedVLMWorkerImpl::init_model(torch::ScalarType dtype,
                                    const ModelArgs& model_args,
                                    const QuantArgs& quant_args) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // initialize model
  ModelArgs tmp_model_args = model_args;
  // TODO refactor later
  tmp_model_args.image_embedding_mode() = true;
  context_.set_model_args(tmp_model_args);

  context_.set_quant_args(quant_args);

  dtype_ = dtype;
  context_.set_tensor_options(torch::dtype(dtype_).device(device_));

  model_ = create_vlm_model(context_);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), tmp_model_args, device_, options_);

  return true;
}

std::optional<ForwardOutput> EmbedVLMWorkerImpl::step(
    const ForwardInput& inputs) {
  torch::DeviceGuard device_guard(device_);
#if defined(USE_NPU)
  torch::npu::synchronize();
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu synchronize stream
#endif

  Timer timer;

  // all tensors should be on the same device as model
  auto flatten_tokens = inputs.token_ids.to(device_);
  auto flatten_positions = inputs.positions.to(device_);
  auto params = inputs.input_params.to(device_);
  auto sampling_params = inputs.sampling_params.to(device_, dtype_);

  // call model executor forward to get hidden states
  auto hidden_states = model_executor_->forward(
      flatten_tokens, flatten_positions, kv_caches_, params);

#if defined(USE_NPU)
  torch::npu::synchronize();
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu synchronize stream
#endif
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!driver_) {
    return std::nullopt;
  }

  // driver prepare model output
  ForwardOutput output;
  output.embedding = hidden_states;
  return output;
}

}  // namespace xllm
