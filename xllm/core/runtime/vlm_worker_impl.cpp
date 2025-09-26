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

#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

VLMWorkerImpl::VLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool VLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  int currentDevId = device_.index();
#if defined(USE_NPU)
  int ret = aclrtSetDevice(currentDevId);
  if (ret != 0) {
    LOG(ERROR) << "ACL set device id:" << currentDevId
               << " failed, ret:" << ret;
  }
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu set device
#endif

  // initialize model
  context.set_image_embedding_mode(false);
  model_ = create_vlm_model(context);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);
  return true;
}

std::optional<ForwardOutput> VLMWorkerImpl::step(
    const BatchedForwardInputs& inputs) {
#if defined(USE_NPU)
  c10_npu::SetDevice(device_.index());
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu set device
#endif
  Timer timer;
  // TODO guojinrong, to adapt multi stream parallel later
  // all tensors should be on the same device as model
  auto flatten_tokens = inputs.micro_inputs[0].token_ids.to(device_);
  auto flatten_positions = inputs.micro_inputs[0].positions.to(device_);
  auto params = inputs.micro_inputs[0].input_params.to(device_);
  auto sampling_params =
      inputs.micro_inputs[0].sampling_params.to(device_, dtype_);

  // call model executor forward to get hidden states
  auto hidden_states = model_executor_->forward(
      {flatten_tokens}, {flatten_positions}, kv_caches_, {params});

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits =
        model_->logits(hidden_states, sampling_params.selected_token_idxes);
  }

#if defined(USE_NPU)
  torch::npu::synchronize();
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu synchronize stream
#endif
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!driver_) {
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
  return output;
}

}  // namespace xllm
