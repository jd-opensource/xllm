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

#define UNIFY_CLASS_NAME(origin_name, target_name) \
  namespace xllm {                                 \
  namespace layer {                                \
  using target_name = origin_name;                 \
  }                                                \
  }

#if defined(USE_NPU)
#include "layers/npu/npu_lm_head_impl.h"
#else
#include "linear.h"
UNIFY_CLASS_NAME(ColumnParallelLinearImpl, LmHeadImpl)
#endif

namespace xllm {
namespace layer {

class LmHead : public torch::nn::ModuleHolder<LmHeadImpl> {
 public:
  using torch::nn::ModuleHolder<LmHeadImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = LmHeadImpl;

  LmHead(const ModelContext& context)
      : ModuleHolder(std::make_shared<LmHeadImpl>(context)) {}
};

}  // namespace layer
}  // namespace xllm
