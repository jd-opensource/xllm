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

#include <vector>

namespace xllm::kernel::npu {

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

void select_unshared_kv(const torch::Tensor& beam_index,
                        const std::vector<torch::Tensor>& x_key_block,
                        const std::vector<torch::Tensor>& x_value_block,
                        const torch::Tensor& block_table,
                        const torch::Tensor& group_offset,
                        int64_t decode_step,
                        int64_t beam_size,
                        int64_t layer_num);

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
}  // namespace xllm::kernel::npu
