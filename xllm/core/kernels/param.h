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

namespace xllm::kernel {

// Note: add default values for optional parameters in the struct definition

// Rotary embedding parameters
struct RotaryParams {
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor sin;
  torch::Tensor cos;
  torch::Tensor cos_sin;
  std::optional<torch::Tensor> position_ids;
  torch::Tensor cu_query_lens;
  bool interleaved;
  bool discrete;
  bool dynamic_ntk = false;
  int max_query_len;
};

// Activation parameters
struct ActivationParams {
  torch::Tensor input;
  torch::Tensor output;
  std::optional<torch::Tensor> bias;
  std::optional<torch::Tensor> cusum_token_count;
  std::string act_mode;
  bool is_gated;
  int start_expert_id = 0;
  int expert_size = 0;
};

// Reshape paged cache parameters
struct ReshapePagedCacheParams {
  torch::Tensor key;
  std::optional<torch::Tensor> value;
  torch::Tensor k_cache;
  std::optional<torch::Tensor> v_cache;
  torch::Tensor slot_mapping;
  bool direction = false;
};

// Attention parameters
struct AttentionParams {
  // common parameters
  torch::Tensor query;
  torch::Tensor output;
  std::optional<torch::Tensor> output_lse;
  std::optional<torch::Tensor> alibi_slope;
  std::optional<torch::Tensor> q_quant_scale;
  std::optional<torch::Tensor> out_quant_scale;
  std::optional<torch::Tensor> block_table;
  std::string compute_dtype;
  int max_seq_len;
  int window_size_left;
  int window_size_right = -1;
  float scale;
  bool return_lse = false;
  // for flashinfer
  torch::Tensor paged_kv_indptr;
  torch::Tensor paged_kv_indices;
  torch::Tensor paged_kv_last_page_len;
  torch::Tensor float_workspace_buffer;
  torch::Tensor int_workspace_buffer;
  torch::Tensor page_locked_int_workspace_buffer;
  torch::Tensor kv_cu_seq_lens;
  torch::Tensor q_cu_seq_lens;
  bool enable_cuda_graph = false;

  // prefill parameters
  torch::Tensor key;    // [num_tokens, num_kv_heads, head_dim_qk]
  torch::Tensor value;  // [num_tokens, num_kv_heads, head_dim_vo]
  std::optional<torch::Tensor> query_start_loc;
  std::optional<torch::Tensor> seq_start_loc;
  std::optional<torch::Tensor> attn_bias;
  std::optional<torch::Tensor> k_quant_scale;
  std::optional<torch::Tensor> v_quant_scale;
  int max_query_len;
  bool is_causal = true;

  // decode parameters
  torch::Tensor k_cache;
  torch::Tensor v_cache;
  torch::Tensor kv_seq_lens;
  std::optional<torch::Tensor> k_cache_quant_scale;
  std::optional<torch::Tensor> v_cache_quant_scale;
  std::optional<torch::Tensor> mask;
  int kv_cache_quant_bit_size = -1;
};

// Fused layer norm parameters
struct FusedLayerNormParams {
  torch::Tensor input;
  torch::Tensor output;
  std::optional<torch::Tensor> residual;
  torch::Tensor weight;
  std::optional<torch::Tensor> beta;
  std::optional<torch::Tensor> bias;
  std::optional<torch::Tensor> quant_scale;
  std::optional<torch::Tensor> residual_out;
  std::optional<torch::Tensor> smooth_quant_scale;
  std::optional<torch::Tensor> normed_out;
  std::string mode;
  double eps;
  bool store_output_before_norm = false;
  bool store_output_after_norm = false;
  bool dynamic_quant = false;
};

// Matmul parameters
struct MatmulParams {
  torch::Tensor a;
  torch::Tensor b;
  std::optional<torch::Tensor> bias;
  std::optional<torch::Tensor> c;
  double alpha = 1.0;
  double beta = 0.0;
};

// Fused MoE parameters
struct FusedMoEParams {
  torch::Tensor hidden_states;
  torch::Tensor gating_output;
  torch::Tensor w1;
  torch::Tensor w2;
  std::optional<torch::Tensor> bias1;
  std::optional<torch::Tensor> bias2;
  std::optional<torch::Tensor> residual;
  std::optional<torch::Tensor> input_smooth;
  std::optional<torch::Tensor> act_smooth;
  std::optional<torch::Tensor> w1_scale;
  std::optional<torch::Tensor> w2_scale;
  std::optional<torch::Tensor> e_score_correction_bias;
  int topk;
  bool renormalize;
  bool gated;
  std::string act_mode;
  std::string scoring_func = "softmax";
  int num_expert_group = -1;
  int topk_group = 0;
  double route_scale = 1.0;
  int start_expert_id = 0;
  int block_n = 0;
  bool avg_moe = false;
  std::optional<torch::Tensor> class_reduce_weight;
  std::optional<torch::Tensor> class_expert_id;
  std::optional<torch::List<int64_t>> w1_quant_flag;
  std::optional<torch::List<int64_t>> w2_quant_flag;
  int world_size = 0;
  int shared_expert_num = 0;
  std::string parallel_mode = "ep";
};

// Per token smooth quantize parameters
struct ScaledQuantizeParams {
  torch::Tensor x;
  torch::Tensor smooth;
  std::optional<torch::Tensor> zero = std::nullopt;
  std::optional<torch::Tensor> token_count = std::nullopt;
  std::optional<torch::Tensor> gather_index = std::nullopt;
  std::optional<torch::Tensor> gather_index_start_position = std::nullopt;
  std::optional<torch::Tensor> output = std::nullopt;
  std::optional<torch::Tensor> output_scale = std::nullopt;
  std::string act_mode = "none";
  double active_coef = 1.0;
  bool is_gated = false;
  torch::ScalarType quant_type = torch::kChar;
};

// Scaled matmul parameters
struct ScaledMatmulParams {
  torch::Tensor a;
  torch::Tensor b;
  std::optional<torch::Tensor> a_scale = std::nullopt;
  torch::Tensor b_scale;
  torch::ScalarType output_dtype;
  std::optional<torch::Tensor> bias = std::nullopt;
  std::optional<torch::Tensor> c = std::nullopt;
  std::string act_mode = "none";
  int64_t quant_bit_size = 8;
  double alpha = 1.0;
  double beta = 1.0;
  bool use_hp_active = false;
  int64_t a_quant_bit_size = -1;
  std::optional<torch::Tensor> a_calib = std::nullopt;
  std::optional<torch::Tensor> b_calib = std::nullopt;
  std::optional<torch::Tensor> output = std::nullopt;
};

struct TopKPParams {
  torch::Tensor logits;
  torch::Tensor temperatures;
  torch::Tensor top_k;
  torch::Tensor top_p;
};

struct RandomSampleParams {
  torch::Tensor logits;
};

// Masked indexer select paged kv parameters
struct MaskedIndexerSelectPagedKVParams {
  bool is_prefill;
  torch::Tensor query;
  torch::Tensor cu_seq_q_lens;
  torch::Tensor cu_seq_k_lens;
  torch::Tensor q_scale;
  torch::Tensor weights;
  double softmax_scale;
  torch::Tensor k_cache;
  torch::Tensor k_context_lens;
  torch::Tensor k_cache_block_table;
  torch::Tensor k_scale_cache;
  int64_t index_topk;
  torch::Tensor kv_cache_block_table;
  int64_t kv_cache_block_size;
  torch::Tensor new_block_table;
  torch::Tensor new_context_lens;
  int64_t quant_block_size;
};

}  // namespace xllm::kernel
