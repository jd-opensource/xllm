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

#include <torch_npu/csrc/aten/CustomFunctions.h>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> apply_moe_gating_topk_softmax(const torch::Tensor& x,
                                                            const std::optional<torch::Tensor>& finished,
                                                            int k) {
  return at_npu::native::custom_ops::npu_moe_gating_top_k_softmax(x, finished.value(), k);
}

}
