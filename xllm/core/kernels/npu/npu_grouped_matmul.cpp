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

#include <vector>

#include "npu_ops_api.h"
#include "ops_npu/npu_ops.h"

namespace xllm::kernel::npu {
namespace {

bool should_use_extended_grouped_matmul_signature(
    const std::optional<torch::TensorList>& bias,
    const std::optional<torch::TensorList>& scale,
    const std::optional<torch::TensorList>& offset,
    const std::optional<torch::TensorList>& antiquant_scale,
    const std::optional<torch::TensorList>& antiquant_offset,
    const std::optional<torch::TensorList>& per_token_scale,
    const std::optional<torch::TensorList>& activation_input,
    const std::optional<torch::TensorList>& activation_quant_scale,
    const std::optional<torch::TensorList>& activation_quant_offset,
    std::optional<int64_t> act_type,
    const c10::OptionalIntArrayRef tuning_config,
    std::optional<torch::ScalarType> output_dtype) {
  return bias.has_value() || scale.has_value() || offset.has_value() ||
         antiquant_scale.has_value() || antiquant_offset.has_value() ||
         per_token_scale.has_value() || activation_input.has_value() ||
         activation_quant_scale.has_value() ||
         activation_quant_offset.has_value() || act_type.has_value() ||
         tuning_config.has_value() || output_dtype.has_value();
}

}  // namespace

std::vector<torch::Tensor> apply_npu_grouped_matmul(
    const torch::TensorList x,
    const torch::TensorList weight,
    const std::optional<torch::TensorList> bias,
    const std::optional<torch::TensorList> scale,
    const std::optional<torch::TensorList> offset,
    const std::optional<torch::TensorList> antiquant_scale,
    const std::optional<torch::TensorList> antiquant_offset,
    const std::optional<torch::TensorList> per_token_scale,
    const std::optional<torch::Tensor>& group_list,
    const std::optional<torch::TensorList> activation_input,
    const std::optional<torch::TensorList> activation_quant_scale,
    const std::optional<torch::TensorList> activation_quant_offset,
    std::optional<int64_t> split_item,
    std::optional<int64_t> group_type,
    std::optional<int64_t> group_list_type,
    std::optional<int64_t> act_type,
    const c10::OptionalIntArrayRef tuning_config,
    std::optional<torch::ScalarType> output_dtype) {
  CHECK(!x.empty()) << "x must not be empty for npu_grouped_matmul.";
  CHECK(group_list.has_value())
      << "group_list is required for npu_grouped_matmul.";

  if (!should_use_extended_grouped_matmul_signature(bias,
                                                    scale,
                                                    offset,
                                                    antiquant_scale,
                                                    antiquant_offset,
                                                    per_token_scale,
                                                    activation_input,
                                                    activation_quant_scale,
                                                    activation_quant_offset,
                                                    act_type,
                                                    tuning_config,
                                                    output_dtype)) {
    const int64_t resolved_split_item = split_item.value_or(2);
    const int64_t resolved_group_type = group_type.value_or(0);
    const int64_t resolved_group_list_type = group_list_type.value_or(1);
    return at_npu::native::custom_ops::npu_grouped_matmul(
        x,
        weight,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        group_list.value(),
        c10::nullopt,
        c10::nullopt,
        c10::nullopt,
        resolved_split_item,
        resolved_group_type,
        resolved_group_list_type);
  }

  const int64_t resolved_split_item = split_item.value_or(2);
  const int64_t resolved_group_type = group_type.value_or(0);
  const int64_t resolved_group_list_type = group_list_type.value_or(1);
  const int64_t resolved_act_type = act_type.value_or(0);
  const auto resolved_tuning_config =
      tuning_config.has_value() ? tuning_config.value() : c10::IntArrayRef();
  const at::ScalarType resolved_output_dtype =
      output_dtype.value_or(x.front().scalar_type());

  return at_npu::native::custom_ops::npu_grouped_matmul(
      x,
      weight,
      bias,
      scale,
      offset,
      antiquant_scale,
      antiquant_offset,
      per_token_scale,
      group_list,
      activation_input,
      activation_quant_scale,
      activation_quant_offset,
      resolved_split_item,
      resolved_group_type,
      resolved_group_list_type,
      resolved_act_type,
      resolved_tuning_config,
      ::std::optional<at::ScalarType>{resolved_output_dtype});
}

}  // namespace xllm::kernel::npu
