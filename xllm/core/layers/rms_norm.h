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

#pragma once
#if defined(USE_NPU)
#include "npu/npu_rms_norm_impl.h"
#elif defined(USE_MLU)
#include "mlu/fuse_norm.h"
#endif

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class RmsNorm : public torch::nn::ModuleHolder<NpuRmsNormImpl> {
 public:
  using torch::nn::ModuleHolder<NpuRmsNormImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuRmsNormImpl;

  RmsNorm(const ModelContext& context)
      : ModuleHolder(std::make_shared<NpuRmsNormImpl>(context)) {}
};
#elif defined(USE_MLU)
class RmsNorm : public torch::nn::ModuleHolder<FusedRMSNormImpl> {
 public:
  using torch::nn::ModuleHolder<FusedRMSNormImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = FusedRMSNormImpl;

  RmsNorm(int64_t dim, double eps, const torch::TensorOptions& options)
      : ModuleHolder(std::make_shared<FusedRMSNormImpl>(dim, eps, options)) {}
};
#endif

}  // namespace layer
}  // namespace xllm
