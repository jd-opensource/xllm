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

namespace xllm::kernel::mlu {
std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    const torch::Tensor& x,
    const torch::Tensor& smooth,
    const std::optional<torch::Tensor>& zero /* = std::nullopt */,
    const std::optional<torch::Tensor>& token_count /* = std::nullopt */,
    const std::optional<torch::Tensor>& gather_index /* = std::nullopt */,
    const std::optional<torch::Tensor>&
        gather_index_start_position /* = std::nullopt */,
    const std::optional<torch::Tensor>& output /* = std::nullopt */,
    const std::optional<torch::Tensor>& output_scale /* = std::nullopt */,
    const std::string& act_mode /* = "none" */,
    double active_coef /* = 1.0 */,
    bool is_gated /* = false */,
    at::ScalarType quant_type /* = at::kChar */
) {
  // If act_mode is "none", override is_gated to false
  bool gated = is_gated;
  if (act_mode == "none") {
    gated = false;
  }

  // Determine output shape
  auto x_sizes = x.sizes();
  std::vector<int64_t> output_shape(x_sizes.begin(), x_sizes.end());
  std::vector<int64_t> output_scale_shape(x_sizes.begin(), x_sizes.end() - 1);

  // Adjust output shape based on gather_index
  if (gather_index.has_value()) {
    int64_t output_tokens = gather_index.value().size(0);
    output_shape[0] = output_tokens;
    output_scale_shape[0] = output_tokens;
  }

  // Adjust output shape for gated activation
  if (gated) {
    // For gated, output is [..., C//2]
    output_shape.back() = output_shape.back() / 2;
  }

  // Allocate output tensors
  torch::Tensor result_output;
  torch::Tensor result_output_scale;

  if (output.has_value()) {
    result_output = output.value();
  } else {
    result_output = torch::empty(output_shape, x.options().dtype(quant_type));
  }

  if (output_scale.has_value()) {
    result_output_scale = output_scale.value();
  } else {
    result_output_scale =
        torch::empty(output_scale_shape, x.options().dtype(at::kFloat));
  }

  // Call underlying MLU kernel
  tmo::torch_api::scaled_quantize(
      x,
      result_output,
      result_output_scale,
      smooth,
      zero,
      token_count,
      gather_index,
      gather_index_start_position,
      /*scale_upper_bound*/ std::nullopt,
      /*quant_algo=*/std::string("dynamic_per_token"),
      act_mode,
      active_coef,
      gated);

  return std::make_tuple(result_output, result_output_scale);
}

}  // namespace xllm::kernel::mlu
