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

std::vector<torch::Tensor> apply_npu_grouped_matmul(const torch::TensorList x,
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

    if (!bias.has_value()) {
        return at_npu::native::custom_ops::npu_grouped_matmul(
        x, weight, c10::nullopt,c10::nullopt,c10::nullopt,c10::nullopt,c10::nullopt,c10::nullopt,group_list.value(),
        c10::nullopt,c10::nullopt,c10::nullopt,split_item.value(), group_type.value(),
        group_list_type.value()
    );
    } else {
        return at_npu::native::custom_ops::npu_grouped_matmul(
        x, weight, bias.value(), scale.value(), offset.value(), antiquant_scale.value(),
        antiquant_offset.value(), per_token_scale.value(), group_list.value(), activation_input.value(),
        activation_quant_scale.value(), activation_quant_offset.value(), split_item.value(), group_type.value(),
        group_list_type.value(), act_type.value(), tuning_config.value(), output_dtype.value()
    );
    }
}

}
