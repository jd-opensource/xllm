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

#include "common/global_flags.h"
#include "common/metrics.h"
#include "runtime/base_executor_impl.h"
#if defined(USE_NPU)
#include "runtime/acl_graph_executor_impl.h"
#else
#include "runtime/mlu_graph_executor_impl.h"
#endif
#include "runtime/options.h"

namespace xllm {

Executor::Executor(CausalLM* model,
                   const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options) {
#if defined(USE_NPU)
  if (FLAGS_enable_acl_graph && device.is_privateuseone()) {
    LOG(INFO) << "Creating ACL Graph Executor for NPU device";
    impl_ =
        std::make_unique<AclGraphExecutorImpl>(model, args, device, options);
    return;
  }
#elif defined(USE_MLU)
  if (FLAGS_enable_graph) {
    LOG(INFO) << "Creating Graph Executor for MLU device";
    impl_ =
        std::make_unique<MluGraphExecutorImpl>(model, args, device, options);
    return;
  }
#endif
  impl_ = std::make_unique<BaseExecutorImpl>(model, args, device, options);
}

ForwardInput Executor::prepare_inputs(Batch& batch) {
  return impl_->prepare_inputs(batch);
}

void Executor::prepare_dp_metadata(const torch::Tensor& tokens,
                                   const ModelInputParams& params,
                                   const ParallelArgs& parallel_args) {
  impl_->prepare_dp_metadata(tokens, params, parallel_args);
}

torch::Tensor Executor::forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params) {
  return impl_->run(tokens, positions, kv_caches, params);
}

}  // namespace xllm
