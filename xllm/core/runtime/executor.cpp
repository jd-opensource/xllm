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

#include "executor.h"

#include <c10/core/TensorOptions.h>
#include <glog/logging.h>

#include "common/metrics.h"
#include "runtime/npu_executor_impl.h"
#include "runtime/options.h"

namespace xllm {

Executor::Executor(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options) {
  impl_ = std::make_unique<NpuExecutorImpl>(model, args, device, options);
}

ForwardInput Executor::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

torch::Tensor Executor::forward(const std::vector<torch::Tensor>& tokens,
                                const std::vector<torch::Tensor>& positions,
                                std::vector<KVCache>& kv_caches,
                                const std::vector<ModelInputParams>& params) {
  COUNTER_INC(num_model_execution_total_eager);
  return impl_->run(tokens, positions, kv_caches, params);
}

}  // namespace xllm
