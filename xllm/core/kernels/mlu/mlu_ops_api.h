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
#include <string>
#include <vector>

#include "ATen/Tensor.h"
#include "torch_mlu_ops.h"

namespace xllm::kernel::mlu {

void apply_rotary(torch::Tensor& q,
                  torch::Tensor& k,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  const std::optional<torch::Tensor>& position_ids,
                  const std::optional<torch::Tensor>& cu_query_lens,
                  bool interleaved,
                  bool discrete,
                  bool dynamic_ntk,
                  int64_t max_query_len);

void active(const torch::Tensor& input,
            torch::Tensor& output,
            const std::optional<torch::Tensor>& bias,
            const std::optional<torch::Tensor>& cusum_token_count,
            const std::string& act_mode,
            bool is_gated,
            int64_t start_expert_id,
            int64_t expert_size);

void reshape_paged_cache(torch::Tensor& key,
                         const std::optional<torch::Tensor>& value,
                         torch::Tensor& k_cache,
                         const std::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping,
                         bool direction);

void batch_prefill(const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   const std::optional<torch::Tensor>& q_cu_seq_lens,
                   const std::optional<torch::Tensor>& kv_cu_seq_lens,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   const std::optional<torch::Tensor>& out_quant_scale,
                   const std::optional<torch::Tensor>& block_tables,
                   int64_t max_query_len,
                   int64_t max_seq_len,
                   float scale,
                   bool is_causal,
                   int64_t window_size_left,
                   int64_t window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse);

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  torch::Tensor& output,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  const std::optional<torch::Tensor>& v_cache,
                  std::optional<torch::Tensor>& output_lse,
                  const std::optional<torch::Tensor>& q_quant_scale,
                  const std::optional<torch::Tensor>& k_cache_quant_scale,
                  const std::optional<torch::Tensor>& v_cache_quant_scale,
                  const std::optional<torch::Tensor>& out_quant_scale,
                  const std::optional<torch::Tensor>& alibi_slope,
                  const std::optional<torch::Tensor>& mask,
                  const std::string& compute_dtype,
                  int64_t max_seq_len,
                  int64_t window_size_left,
                  int64_t window_size_right,
                  float scale,
                  bool return_lse,
                  int64_t kv_cache_quant_bit_size);

void masked_indexer_select_paged_kv(const bool is_prefill,
                                    const torch::Tensor& query,
                                    const torch::Tensor& cu_seq_q_lens,
                                    const torch::Tensor& cu_seq_k_lens,
                                    const torch::Tensor& q_scale,
                                    const torch::Tensor& weights,
                                    const double softmax_scale,
                                    const torch::Tensor& k_cache,
                                    const torch::Tensor& k_context_lens,
                                    const torch::Tensor& k_cache_block_table,
                                    const torch::Tensor& k_scale_cache,
                                    const int64_t index_topk,
                                    const torch::Tensor& kv_cache_block_table,
                                    const int64_t kv_cache_block_size,
                                    const torch::Tensor& new_block_table,
                                    const torch::Tensor& new_context_lens,
                                    const int64_t quant_block_size);

void fused_layernorm(const torch::Tensor& input,
                     torch::Tensor& output,
                     const std::optional<torch::Tensor>& residual,
                     const torch::Tensor& weight,
                     const std::optional<torch::Tensor>& beta,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& quant_scale,
                     const std::optional<torch::Tensor>& residual_out,
                     const std::optional<torch::Tensor>& smooth_quant_scale,
                     const std::optional<torch::Tensor>& normed_out,
                     const std::string& mode,
                     double eps,
                     bool store_output_before_norm,
                     bool store_output_after_norm,
                     bool dynamic_quant);

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& c,
                     double alpha,
                     double beta);

torch::Tensor group_gemm(const torch::Tensor& a,
                         const torch::Tensor& b,
                         const torch::Tensor& token_count,
                         torch::Tensor& output,
                         const std::optional<torch::Tensor>& a_scale,
                         const std::optional<torch::Tensor>& b_scale,
                         const std::optional<torch::List<int64_t>>& quant_flag,
                         const int64_t max_dim,
                         const bool trans_a,
                         const bool trans_b,
                         const int64_t a_quant_bit);

std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    const torch::Tensor& input,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool normalize,
    const std::optional<torch::Tensor>& mask,
    const std::string& normed_by,
    const std::string& scoring_func,
    double route_scale,
    const std::optional<torch::Tensor>& e_score_correction_bias);

std::vector<torch::Tensor> moe_gen_idx(const torch::Tensor& expert_id,
                                       int64_t expert_num);

torch::Tensor moe_expand_input(
    const torch::Tensor& input,
    const torch::Tensor& gather_index,
    const std::optional<torch::Tensor>& cusum_token_count,
    int64_t start_expert_id,
    int64_t expert_size);

torch::Tensor moe_combine_result(
    const torch::Tensor& input,
    const torch::Tensor& reduce_weight,
    const torch::Tensor& gather_ids,
    const std::optional<torch::Tensor>& residual,
    const std::optional<torch::Tensor>& cusum_token_count,
    const int64_t start_expert_id,
    const int64_t expert_size,
    const std::optional<torch::Tensor>& bias);

torch::Tensor moe_all2all_gen_send_layout(const torch::Tensor& token_count,
                                          int64_t nrank);

std::vector<torch::Tensor> moe_all2all_gen_gather_index(
    const torch::Tensor& token_num,
    int64_t pad_num,
    bool return_cusum_token_count);

std::vector<torch::Tensor> moe_all2all_create(int64_t dispatch_token_byte,
                                              int64_t combine_token_byte,
                                              int64_t max_expert_num,
                                              int64_t max_token_num,
                                              int64_t rank,
                                              int64_t nrank,
                                              const torch::Device& device);

void moe_all2all_init(int64_t handle,
                      const torch::Tensor& all_exchange_info,
                      const torch::Device& device);

void moe_all2all_dispatch(int64_t handle,
                          int64_t token_byte,
                          int64_t token_num,
                          const torch::Tensor& send_layout,
                          const torch::Tensor& send_token_num,
                          const torch::Tensor& recv_layout,
                          const torch::Tensor& recv_token_num,
                          const std::optional<torch::Tensor>& send_token,
                          const std::optional<torch::Tensor>& recv_token);

void moe_all2all_combine(int64_t handle,
                         int64_t token_byte,
                         int64_t token_num,
                         const torch::Tensor& send_src_layout,
                         const torch::Tensor& send_dst_layout,
                         const std::optional<torch::Tensor>& send_token,
                         const std::optional<torch::Tensor>& recv_token);

void moe_all2all_destroy(int64_t handle, const torch::Device& device);

std::tuple<torch::Tensor, torch::Tensor> scaled_quantize(
    const torch::Tensor& x,
    const torch::Tensor& smooth,
    const std::optional<torch::Tensor>& zero = std::nullopt,
    const std::optional<torch::Tensor>& token_count = std::nullopt,
    const std::optional<torch::Tensor>& gather_index = std::nullopt,
    const std::optional<torch::Tensor>& gather_index_start_position =
        std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt,
    const std::optional<torch::Tensor>& output_scale = std::nullopt,
    const std::string& act_mode = "none",
    double active_coef = 1.0,
    bool is_gated = false,
    torch::ScalarType quant_type = torch::kChar);

torch::Tensor scaled_matmul(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const std::optional<torch::Tensor>& a_scale,
    const torch::Tensor& b_scale,
    torch::ScalarType output_dtype,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& c = std::nullopt,
    const std::string& act_mode = "none",
    int64_t quant_bit_size = 8,
    double alpha = 1.0,
    double beta = 1.0,
    bool use_hp_active = false,
    int64_t a_quant_bit_size = -1,
    const std::optional<torch::Tensor>& a_calib = std::nullopt,
    const std::optional<torch::Tensor>& b_calib = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt);

torch::Tensor apply_top_k_top_p(const torch::Tensor& logits,
                                const torch::Tensor& temperature_list,
                                const torch::Tensor& topk_list,
                                const torch::Tensor& topp_list);

torch::Tensor random_sample(const torch::Tensor& probs);

void gather_split(const torch::Tensor& input,
                  const torch::Tensor& gather_index,
                  const torch::Tensor& valid_token_num,
                  const torch::Tensor& output_head,
                  const torch::Tensor& output_tail);

}  // namespace xllm::kernel::mlu
