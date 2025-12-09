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

void residual_layer_norm(torch::Tensor& input,
                         torch::Tensor& output,
                         std::optional<torch::Tensor>& residual,
                         torch::Tensor& weight,
                         std::optional<torch::Tensor>& beta,
                         std::optional<torch::Tensor>& bias,
                         std::optional<torch::Tensor>& residual_out,
                         double eps) {
  torch::ScalarType scalar_type = input.scalar_type();
  int hidden_size = weight.numel();
  torch::Tensor beta_ = beta.value_or(at::zeros(
      {hidden_size},
      at::TensorOptions().dtype(input.scalar_type()).device(input.device())));
  auto residual_ = residual.value_or(torch::zeros_like(input));
  std::optional<torch::Tensor> output_ = output;
  infer::residual_layer_norm(input,
                             residual_,
                             weight,
                             beta_,
                             bias,
                             output_,
                             residual_out,
                             1.0,
                             eps,
                             false);
}
}  // namespace xllm::kernel::ilu