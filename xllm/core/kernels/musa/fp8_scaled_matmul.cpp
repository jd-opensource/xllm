/* Copyright 2025-2026 The xLLM Authors.

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

#include "cuda_ops_api.h"

namespace xllm::kernel::cuda {

torch::Tensor fp8_scaled_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& bias /* = std::nullopt */,
    const std::optional<torch::Tensor>& output /* = std::nullopt */) {
  // Prepare output tensor
  torch::Tensor result_output;
  if (output.has_value() && output.value().defined()) {
    result_output = output.value();
  } else {
    result_output =
        torch::empty({a.size(0), b.size(0)}, a.options().dtype(output_dtype));
  }

  // Transpose weight for CUTLASS: [N, K] -> [K, N] (column-major)
  // NOTE: Do NOT call .contiguous() - .t() makes it column-major (stride(0)==1)
#if defined(XLLM_TORCH_MUSA)
  const auto a_f = a.to(output_dtype) * a_scale;
  const auto b_f = b.to(output_dtype) * b_scale;
  auto out = at::matmul(a_f, b_f.t());
  if (bias.has_value() && bias.value().defined()) {
    out = out + bias.value();
  }
  if (output.has_value() && output.value().defined()) {
    result_output.copy_(out);
  } else {
    result_output = std::move(out);
  }
#else
  auto b_t = b.t();
  cutlass_scaled_mm(result_output, a, b_t, a_scale, b_scale, bias);
#endif

  return result_output;
}

}  // namespace xllm::kernel::cuda
