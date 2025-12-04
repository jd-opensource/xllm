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

#include "ilu_ops_api.h"
#include "utils.h"

using namespace ixformer;

namespace xllm::kernel::ilu {

void layer_norm(at::Tensor& input,
                at::Tensor& weight,
                c10::optional<at::Tensor>& bias_tensor,
                at::Tensor& output,
                double eps) {
  auto scalar_type = input.scalar_type();
  int hidden_size = weight.numel();
  at::Tensor bias = bias_tensor.value_or(at::zeros(
      {hidden_size},
      at::TensorOptions().dtype(input.scalar_type()).device(input.device())));
  std::optional<at::Tensor> fused_bias = std::nullopt;
  infer::layer_norm(input, weight, bias, fused_bias, output, eps);
}
}  // namespace xllm::kernel::ilu