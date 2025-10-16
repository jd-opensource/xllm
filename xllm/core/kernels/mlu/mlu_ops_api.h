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
                  const torch::Tensor& cu_query_lens,
                  bool interleaved,
                  bool discrete,
                  bool dynamic_ntk,
                  int max_query_len);

void active(const torch::Tensor& input,
            torch::Tensor& output,
            const std::optional<torch::Tensor>& bias,
            const std::optional<torch::Tensor>& cusum_token_count,
            const std::string& act_mode,
            bool is_gated,
            int start_expert_id,
            int expert_size);

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
                   const std::optional<torch::Tensor>& query_start_loc,
                   const std::optional<torch::Tensor>& seq_start_loc,
                   const std::optional<torch::Tensor>& alibi_slope,
                   const std::optional<torch::Tensor>& attn_bias,
                   const std::optional<torch::Tensor>& q_quant_scale,
                   const std::optional<torch::Tensor>& k_quant_scale,
                   const std::optional<torch::Tensor>& v_quant_scale,
                   const std::optional<torch::Tensor>& out_quant_scale,
                   const std::optional<torch::Tensor>& block_tables,
                   int max_query_len,
                   int max_seq_len,
                   float scale,
                   bool is_causal,
                   int window_size_left,
                   int window_size_right,
                   const std::string& compute_dtype,
                   bool return_lse);

void batch_decode(const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  torch::Tensor& output,
                  const torch::Tensor& block_table,
                  const torch::Tensor& seq_lens,
                  const torch::Tensor& v_cache,
                  std::optional<torch::Tensor>& output_lse,
                  const std::optional<torch::Tensor>& q_quant_scale,
                  const std::optional<torch::Tensor>& k_cache_quant_scale,
                  const std::optional<torch::Tensor>& v_cache_quant_scale,
                  const std::optional<torch::Tensor>& out_quant_scale,
                  const std::optional<torch::Tensor>& alibi_slope,
                  const std::optional<torch::Tensor>& mask,
                  const std::string& compute_dtype,
                  int max_seq_len,
                  int window_size_left,
                  int window_size_right,
                  float scale,
                  bool return_lse,
                  int kv_cache_quant_bit_size);

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

torch::Tensor fused_moe(
    const torch::Tensor& hidden_states,
    const torch::Tensor& gating_output,
    const torch::Tensor& w1,
    const torch::Tensor& w2,
    const std::optional<torch::Tensor>& bias1,
    const std::optional<torch::Tensor>& bias2,
    const std::optional<torch::Tensor>& residual,
    const std::optional<torch::Tensor>& input_smooth,
    const std::optional<torch::Tensor>& act_smooth,
    const std::optional<torch::Tensor>& w1_scale,
    const std::optional<torch::Tensor>& w2_scale,
    const std::optional<torch::Tensor>& e_score_correction_bias,
    int topk,
    bool renormalize,
    bool gated,
    const std::string& act_mode,
    const std::string& scoring_func,
    int num_expert_group,
    int topk_group,
    double route_scale,
    int start_expert_id,
    int block_n,
    bool avg_moe,
    const std::optional<torch::Tensor>& class_reduce_weight,
    const std::optional<torch::Tensor>& class_expert_id,
    const std::optional<torch::List<int64_t>>& w1_quant_flag,
    const std::optional<torch::List<int64_t>>& w2_quant_flag,
    int world_size,
    int shared_expert_num,
    const std::string& parallel_mode);

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

}  // namespace xllm::kernel::mlu
