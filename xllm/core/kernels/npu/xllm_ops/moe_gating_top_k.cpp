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

#include <torch/library.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

void check_moe_gating_top_k_shape_and_dtype(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& bias,
    int64_t k) {
  TORCH_CHECK(x.dim() == 2,
              "Input tensor x's dim num should be 2, actual ",
              x.dim(),
              ".");
  TORCH_CHECK(x.size(1) > 0,
              "Input tensor x's expert dim should be positive, actual ",
              x.size(1),
              ".");
  TORCH_CHECK(k > 0, "Attribute k should be greater than 0, actual ", k, ".");
  TORCH_CHECK(k <= x.size(1),
              "Attribute k should be no greater than x.shape[-1], actual k is ",
              k,
              ", x.shape[-1] is ",
              x.size(1),
              ".");
  TORCH_CHECK(x.dtype() == at::kFloat || x.dtype() == at::kHalf ||
                  x.dtype() == at::kBFloat16,
              "x should be FLOAT16, BFLOAT16, or FLOAT32.");

  if (bias.has_value()) {
    const at::Tensor& bias_tensor = bias.value();
    TORCH_CHECK(bias_tensor.dtype() == x.dtype(),
                "bias's dtype should be equal to x's dtype.");
    TORCH_CHECK(bias_tensor.dim() == 1,
                "bias's dim num should be 1, actual ",
                bias_tensor.dim(),
                ".");
    TORCH_CHECK(bias_tensor.size(0) == x.size(1),
                "bias's first dim should be equal to x's expert dim.");
  }
}

c10::optional<at::Tensor> defined_tensor_or_nullopt(
    const c10::optional<at::Tensor>& tensor) {
  if (tensor.has_value() && tensor->defined()) {
    return tensor.value();
  }
  return c10::nullopt;
}

at::Tensor construct_moe_gating_top_k_y_tensor(const at::Tensor& x, int64_t k) {
  return at::empty({x.size(0), k}, x.options().dtype(x.dtype()));
}

at::Tensor construct_moe_gating_top_k_expert_idx_tensor(const at::Tensor& y) {
  return at::empty(y.sizes(), y.options().dtype(at::kInt));
}

at::Tensor construct_moe_gating_top_k_out_tensor(const at::Tensor& x) {
  return at::empty(x.sizes(), x.options().dtype(at::kFloat));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_gating_top_k(
    const at::Tensor& x,
    int64_t k,
    const c10::optional<at::Tensor>& bias,
    int64_t k_group,
    int64_t group_count,
    double routed_scaling_factor,
    double eps,
    int64_t group_select_mode,
    int64_t renorm,
    int64_t norm_type,
    bool out_flag) {
  const c10::optional<at::Tensor> bias_opt = defined_tensor_or_nullopt(bias);
  check_moe_gating_top_k_shape_and_dtype(x, bias_opt, k);
  at::Tensor y = construct_moe_gating_top_k_y_tensor(x, k);
  at::Tensor expert_idx = construct_moe_gating_top_k_expert_idx_tensor(y);
  at::Tensor out = construct_moe_gating_top_k_out_tensor(x);

  EXEC_NPU_CMD(aclnnMoeGatingTopK,
               x,
               bias_opt,
               k,
               k_group,
               group_count,
               group_select_mode,
               renorm,
               norm_type,
               out_flag,
               routed_scaling_factor,
               eps,
               y,
               expert_idx,
               out);
  return std::make_tuple(y, expert_idx, out);
}

}  // namespace xllm::kernel::npu
