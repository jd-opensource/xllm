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
#include "impl/npu_split_impl.h"

namespace xllm::kernel {
class Split : public torch::nn::ModuleHolder<NpuSplitImpl> {
 public:
  using torch::nn::ModuleHolder<NpuSplitImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = NpuSplitImpl;

  Split(const ModelContext& context,
        int32_t splitDim = 2,
        int32_t splitNum = 3,
        atb::SVector<int32_t> splitSizes = {})
      : ModuleHolder(std::make_shared<NpuSplitImpl>(context,
                                                    splitDim,
                                                    splitNum,
                                                    splitSizes)) {}
};

}  // namespace xllm::kernel
