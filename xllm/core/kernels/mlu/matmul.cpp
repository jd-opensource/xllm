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

#include "mlu_ops_api.h"
#include "torch_mlu_ops.h"

namespace xllm::kernel::mlu {

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& c,
                     double alpha,
                     double beta) {
  return tmo::torch_api::matmul(a,
                                b,
                                bias,
                                c,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                std::nullopt,
                                "none",
                                alpha,
                                beta,
                                true,
                                true,
                                1.0,
                                1.0,
                                false,
                                true);
}

}  // namespace xllm::kernel::mlu
