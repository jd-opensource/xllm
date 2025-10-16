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

namespace xllm::mlu {

static const std::string kActModeSilu = "silu";
static const std::string kActModeGelu = "gelu";
static const std::string kActModeQuickGelu = "quick_gelu";
static const std::string kActModeSwish = "swish";

void apply_rotary(const torch::Tensor& input,
                  torch::Tensor& output,
                  const torch::Tensor& sin,
                  const torch::Tensor& cos,
                  const std::optional<torch::Tensor>& position_ids,
                  const torch::Tensor& cu_query_lens,
                  bool interleaved,
                  bool discrete,
                  bool dynamic_ntk = false,
                  int max_query_len);

void active(
    const torch::Tensor& input,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& cusum_token_count = std::nullopt,
    const std::string& act_mode,
    bool is_gated,
    int start_expert_id = 0,
    int expert_size = 0);

void reshape_paged_cache(const torch::Tensor& key,
                         const torch::Tensor& value,
                         torch::Tensor& k_cache,
                         torch::Tensor& v_cache,
                         const torch::Tensor& slot_mapping,
                         bool direction = false);

void flash_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& output,
    torch::Tensor& output_lse,
    int query_start_loc,
    int seq_start_loc,
    const std::optional<torch::Tensor>& alibi_slope = std::nullopt,
    const std::optional<torch::Tensor>& attn_bias = std::nullopt,
    const std::optional<torch::Tensor>& q_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& k_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& v_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& out_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& block_tables = std::nullopt,
    int max_query_len,
    int max_seq_len,
    float scale,
    bool is_causal = true,
    int window_size_left,
    int window_size_right = -1,
    const std::string& compute_dtype,
    bool return_lse = false);

void single_query_cached_kv_attn(
    const torch::Tensor& query,
    const torch::Tensor& k_cache,
    torch::Tensor& output,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    const torch::Tensor& v_cache,
    torch::Tensor& output_lse,
    const std::optional<torch::Tensor>& q_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& k_cache_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& v_cache_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& out_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& alibi_slope = std::nullopt,
    const std::optional<torch::Tensor>& mask = std::nullopt,
    const std::string& compute_dtype,
    int max_seq_len,
    int window_size_left,
    int window_size_right = -1,
    float scale,
    bool return_lse = false,
    int kv_cache_quant_bit_size = -1);

void fused_layernorm(
    const torch::Tensor& input,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& residual = std::nullopt,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& beta = std::nullopt,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    const std::optional<torch::Tensor>& quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& residual_out,
    const std::optional<torch::Tensor>& smooth_quant_scale = std::nullopt,
    const std::optional<torch::Tensor>& normed_out = std::nullopt,
    const std::string& mode,
    double eps,
    bool store_output_before_norm = false,
    bool store_output_after_norm = false,
    bool dynamic_quant = false);

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
    int topk,
    bool renormalize,
    bool gated,
    const std::string& act_mode,
    const std::string& scoring_func = "softmax",
    int start_expert_id = 0,
    int block_n = 0,
    bool avg_moe = false,
    const std::optional<torch::Tensor>& class_reduce_weight = std::nullopt,
    const std::optional<torch::Tensor>& class_expert_id = std::nullopt,
    const std::optional<std::vector<bool>>& w1_quant_flag = std::nullopt,
    const std::optional<std::vector<bool>>& w2_quant_flag = std::nullopt,
    int world_size = 0,
    int shared_expert_num = 0,
    const std::string& parallel_mode = "ep");

}  // namespace xllm::mlu
