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

#include "linear.h"

namespace xllm {
namespace layer {

class LmHead : public torch::nn::ModuleHolder<RowParallelLinearImpl> {
 public:
  using torch::nn::ModuleHolder<RowParallelLinearImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = RowParallelLinearImpl;

  LmHead(const ModelContext& context)
      : ModuleHolder(std::make_shared<RowParallelLinearImpl>(
            // NOTE: Quantization should NOT be used for the final language
            // modeling head (lm_head). The output logits must remain in high
            // precision (typically bfloat16/float16) for numerical stability
            // and correct evaluation of loss and predictions. Always use
            // unquantized weights here.
            context.get_model_args().hidden_size(),
            context.get_model_args().vocab_size(),
            /*bias=*/false,
            /*input_is_parallelized=*/false,
            /*enable_result_reduction=*/true,
            QuantArgs{},  // do not use quantization for lm_head!
            context.get_parallel_args().tp_group_,
            context.get_tensor_options())) {}
};

}  // namespace layer
}  // namespace xllm
