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

#pragma once
#include <torch/torch.h>

#include <optional>
#include <tuple>

#include "custom_functions_npu/atb_common.h"

namespace xllm::kernel::npu {

void reshape_paged_cache(torch::Tensor& key,
                         std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping);

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   const torch::Tensor& mask,
                   const torch::Tensor& seq_len,
                   float scale,
                   torch::Tensor& output);

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  const torch::Tensor& v_cache,
                  float scale,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  torch::Tensor& output);

// Custom batch decode for ACL graph execution
// This variant uses CustomPagedAttention to avoid .to(kCPU) operations
// that break ACL graph capture
void batch_decode_acl_graph(const torch::Tensor& query,
                            const torch::Tensor& k_cache,
                            const torch::Tensor& v_cache,
                            float scale,
                            const torch::Tensor& block_table,
                            const torch::Tensor& seq_lens,
                            const torch::Tensor& tiling_data,
                            torch::Tensor& output);

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias);

torch::Tensor active(const torch::Tensor& input, const std::string& act_mode);

torch::Tensor rms_norm(const torch::Tensor& input,
                       const torch::Tensor& weight,
                       double eps,
                       const std::string& mode);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> add_rms_norm(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    const torch::Tensor& gamma,
    double epsilon);

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& cos_sin_cache,
                  const torch::Tensor& positions);

torch::Tensor apply_npu_moe_token_unpermute(const torch::Tensor& permuted_tokens,
                                            const torch::Tensor& sorted_indices,
                                            const std::optional<torch::Tensor>& probes,
                                            bool padded_mode,
                                            c10::OptionalIntArrayRef restore_shape);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> apply_moe_gating_topk_softmax(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& finished,
    int k);

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
                                                    std::optional<torch::ScalarType> output_dtype);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> apply_npu_moe_init_routing_v2(
    const torch::Tensor& x,
    const torch::Tensor& expert_idx,
    const std::optional<torch::Tensor>& scale,
    const std::optional<torch::Tensor>& offset,
    int active_num,
    int expert_capacity,
    int expert_num,
    int drop_pad_mode,
    int expert_tokens_num_type,
    bool expert_tokens_num_flag,
    int quant_mode,
    torch::IntArrayRef active_expert_range,
    int row_idx_type);

}  // namespace xllm::kernel::npu
