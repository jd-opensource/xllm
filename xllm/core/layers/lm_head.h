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
#include "npu/npu_lm_head_impl.h"
#endif
#include "common/linear_impl.h"

namespace xllm {
namespace layer {

#if defined(USE_NPU)
class NpuLmHead : public torch::nn::ModuleHolder<NpuLmHeadImpl> {
 public:
  using torch::nn::ModuleHolder<NpuLmHeadImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuLmHeadImpl;

  NpuLmHead(const ModelContext& context)
      : ModuleHolder(std::make_shared<NpuLmHeadImpl>(context)) {}
};

#endif
class LmHead : public torch::nn::ModuleHolder<ColumnParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<ColumnParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = ColumnParallelLinearImpl;

  LmHead(int64_t in_features,
         int64_t out_features,
         bool bias,
         bool gather_output,
         const QuantArgs& quant_args,
         const ParallelArgs& parallel_args,
         const torch::TensorOptions& options)
      : ModuleHolder(std::make_shared<ColumnParallelLinearImpl>(in_features,
                                                                out_features,
                                                                bias,
                                                                gather_output,
                                                                quant_args,
                                                                parallel_args,
                                                                options)) {}
};

}  // namespace layer
}  // namespace xllm