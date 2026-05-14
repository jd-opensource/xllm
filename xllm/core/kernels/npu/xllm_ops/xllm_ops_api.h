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

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace xllm::kernel::npu {
namespace op_infer {
constexpr int32_t N = 32;
// npu tensor max size
constexpr int32_t SIZE = 8;
constexpr int32_t INT4_NUMS_IN_INT32_SPACE = 8;
constexpr int32_t NPU_NSA_COMPRESS_INPUT_DIM_SECOND = 1;
constexpr int32_t NPU_NSA_COMPRESS_INPUT_DIM_THIRD = 2;
constexpr int32_t DIM_0 = 0;
constexpr int32_t DIM_1 = 1;
constexpr int32_t DIM_2 = 2;
constexpr int32_t DIM_3 = 3;
}  // namespace op_infer

void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& src_seq_idxes,
                 torch::Tensor& out_logprobs,
                 torch::Tensor& out_token_ids);

void top_k_top_p(torch::Tensor& logits,
                 const torch::Tensor& topK,
                 const torch::Tensor& topP);

void replace_token(torch::Tensor& dst, torch::Tensor& src);

void beam_search_rec(const torch::Tensor& logprobs,
                     const torch::Tensor& top_tokens,
                     const torch::Tensor& top_logprobs,
                     torch::Tensor& sequence_group,
                     int64_t current_step,
                     torch::Tensor& out_token_ids,
                     torch::Tensor& out_token_index,
                     torch::Tensor& out_log_probs,
                     torch::Tensor& out_beam_count_prefix_sums,
                     torch::Tensor& out_sequence);

void beam_search_rec(const torch::Tensor& logprobs,
                     const torch::Tensor& top_tokens,
                     const torch::Tensor& top_logprobs,
                     torch::Tensor& sequence_group,
                     int64_t current_step,
                     int64_t result_width,
                     torch::Tensor& out_token_ids,
                     torch::Tensor& out_token_index,
                     torch::Tensor& out_log_probs,
                     torch::Tensor& out_beam_count_prefix_sums,
                     torch::Tensor& out_sequence);

void select_unshared_kv(const torch::Tensor& beam_index,
                        const std::vector<torch::Tensor>& x_key_block,
                        const std::vector<torch::Tensor>& x_value_block,
                        const torch::Tensor& block_table,
                        const torch::Tensor& group_offset,
                        int64_t decode_step,
                        int64_t beam_size,
                        int64_t layer_num);

std::optional<std::tuple<torch::Tensor, torch::Tensor>>
rec_constrained_topk_fused(const torch::Tensor& logits,
                           const torch::Tensor& sequence_group,
                           const torch::Tensor& first_token_ids,
                           const torch::Tensor& prefix1_offsets,
                           const torch::Tensor& prefix1_values,
                           const torch::Tensor& prefix1_pair_keys,
                           const torch::Tensor& prefix2_value_offsets,
                           const torch::Tensor& prefix2_values,
                           const torch::Tensor& temperatures,
                           int64_t current_step,
                           int64_t top_k,
                           int64_t max_prefix1_degree,
                           int64_t max_prefix2_degree);

torch::Tensor causal_conv1d(const torch::Tensor& x,
                            const torch::Tensor& weight,
                            const torch::Tensor& conv_state,
                            const std::optional<torch::Tensor>& bias_opt,
                            const torch::IntArrayRef query_start_loc_opt,
                            const torch::IntArrayRef cache_indices_opt,
                            const torch::IntArrayRef initial_state_mode_opt,
                            const torch::IntArrayRef num_accepted_tokens_opt,
                            int64_t activation_mode,
                            int64_t pad_slot_id,
                            int64_t run_mode);

at::Tensor quant_matmul(const at::Tensor& x1,
                        const at::Tensor& x2,
                        const bool transpose2,
                        const at::Tensor& scale,
                        const c10::optional<at::Tensor>& offset,
                        const c10::optional<at::Tensor>& pertoken_scale,
                        const c10::optional<at::Tensor>& bias,
                        c10::optional<at::ScalarType> output_dtype);

at::Tensor quantize_per_tensor(const at::Tensor& self,
                               const at::Tensor& scales,
                               const at::Tensor& zero_points,
                               at::ScalarType dtype,
                               int64_t axis);

std::tuple<at::Tensor, c10::optional<at::Tensor>> dynamic_quant(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& smooth_scales,
    const c10::optional<at::Tensor>& group_index,
    c10::optional<at::ScalarType> dst_type);

std::tuple<at::Tensor, at::Tensor> dequant_swiglu_quant(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& weight_scale,
    const c10::optional<at::Tensor>& activation_scale,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& quant_scale,
    const c10::optional<at::Tensor>& quant_offset,
    const c10::optional<at::Tensor>& group_index,
    bool activate_left,
    int64_t quant_mode);
}  // namespace xllm::kernel::npu
