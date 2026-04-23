/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

namespace xllm::kernel::mlu {
torch::Tensor gated_layer_norm(torch::Tensor& x,
                               const torch::Tensor& weight,
                               const torch::Tensor& bias,
                               double eps,
                               const std::optional<torch::Tensor>& gate,
                               int64_t group_size,
                               bool norm_before_gate) {
  if (gate.has_value() && !norm_before_gate) {
    torch::Tensor gate_value = gate.value();
    gate_value = gate_value.to(torch::kFloat32);
    x = x * torch::silu(gate_value);
  }

  int64_t hidden_size = x.size(-1);
  torch::Tensor normalized;
  if (group_size == hidden_size) {
    auto variance = x.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    normalized = x * torch::rsqrt(variance + eps);
    normalized = normalized * weight;
  } else {
    int64_t num_groups = hidden_size / group_size;
    auto x_grouped = x.reshape({-1, num_groups, group_size});
    auto variance = x_grouped.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    auto normalized_grouped = x_grouped * torch::rsqrt(variance + eps);
    normalized = normalized_grouped.reshape({-1, num_groups * group_size});
    normalized = normalized * weight;
  }

  if (gate.has_value() && norm_before_gate) {
    auto gate_value = gate.value().to(torch::kFloat32);
    normalized = normalized * torch::silu(gate_value);
  }

  return normalized;
}
}  // namespace xllm::kernel::mlu
