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
#include "util/env_var.h"

namespace xllm::kernel::ilu {

bool gemv_conditions(const torch::Tensor& input,
                     const torch::Tensor& weight,
                     const torch::Tensor& bias,
                     int64_t gemv_max_batch) {
  // gemv input:[m,k] weight:[n,k]
  // 1. m <= gemv_max_batch
  // 2. k % 32 == 0 && n % 2 == 0
  // 3. bias is None

  torch::Tensor input_view = input.view({-1, input.size(-1)});
  torch::Tensor weight_view = weight.view({-1, weight.size(-1)});

  int64_t m = input_view.size(0);
  int64_t k = input_view.size(1);
  int64_t n = weight_view.size(0);

  if (bias.defined() == false && m <= gemv_max_batch && k % 32 == 0 &&
      n % 2 == 0) {
    return true;
  }
  return false;
}

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias) {
  int64_t act_type = -1;
  bool persistent = false;
  std::vector<int64_t> output_shape = a.sizes().vec();
  if (!output_shape.empty()) {
    output_shape[output_shape.size() - 1] = b.size(0);
  }
  torch::Tensor output = a.new_empty(output_shape);

  bool use_gemv = true;
  const int64_t gemv_max_batch = 1;
  const char* disable_infer_gemm_ex_str =
      get_string_env("DISABLE_INFER_GEMM_EX");
  std::string disable_infer_gemm_ex =
      (disable_infer_gemm_ex_str == nullptr) ? "0" : disable_infer_gemm_ex_str;

  use_gemv =
      use_gemv &&
      gemv_conditions(a, b, bias.value_or(at::Tensor()), gemv_max_batch) &&
      (disable_infer_gemm_ex != "1") && (act_type == -1);

  if (use_gemv) {
    output = infer::ixformer_linear_ex(a, b, bias, output);
  } else {
    output = infer::ixformer_linear(a, b, act_type, bias, output, persistent);
  }
  return output;
}

}  // namespace xllm::kernel::ilu