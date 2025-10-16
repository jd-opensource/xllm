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

namespace xllm {
namespace kernel {

// Rotary embedding parameters
struct RotaryParams {
  torch::Tensor input;
  torch::Tensor output;
  torch::Tensor sin;
  torch::Tensor cos;
  std::optional<torch::Tensor> position_ids;
  torch::Tensor cu_query_lens;
  bool interleaved;
  bool discrete;
  bool dynamic_ntk;
  int max_query_len;

  RotaryParams() : position_ids(std::nullopt), dynamic_ntk(false) {}
};

// Activation parameters
struct ActivationParams {
  torch::Tensor input;
  torch::Tensor output;
  std::optional<torch::Tensor> bias;
  std::optional<torch::Tensor> cusum_token_count;
  std::string act_mode;
  bool is_gated;
  int start_expert_id;
  int expert_size;

  ActivationParams()
      : bias(std::nullopt),
        cusum_token_count(std::nullopt),
        start_expert_id(0),
        expert_size(0) {}
};

// Reshape paged cache parameters
struct ReshapePagedCacheParams {
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor k_cache;
  torch::Tensor v_cache;
  torch::Tensor slot_mapping;
  bool direction;

  ReshapePagedCacheParams() : direction(false) {}
};

// Prefill attention parameters
struct PrefillAttentionParams {
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor output;
  torch::Tensor output_lse;
  torch::Tensor query_start_loc;
  torch::Tensor seq_start_loc;
  std::optional<torch::Tensor> alibi_slope;
  std::optional<torch::Tensor> attn_bias;
  std::optional<torch::Tensor> q_quant_scale;
  std::optional<torch::Tensor> k_quant_scale;
  std::optional<torch::Tensor> v_quant_scale;
  std::optional<torch::Tensor> out_quant_scale;
  std::optional<torch::Tensor> block_tables;
  int max_query_len;
  int max_seq_len;
  float scale;
  bool is_causal;
  int window_size_left;
  int window_size_right;
  std::string compute_dtype;
  bool return_lse;

  FlashAttentionParams()
      : alibi_slope(std::nullopt),
        attn_bias(std::nullopt),
        q_quant_scale(std::nullopt),
        k_quant_scale(std::nullopt),
        v_quant_scale(std::nullopt),
        out_quant_scale(std::nullopt),
        block_tables(std::nullopt),
        is_causal(true),
        window_size_right(-1),
        return_lse(false) {}
};

// Decode attention parameters
struct DecodeAttentionParams {
  torch::Tensor query;
  torch::Tensor k_cache;
  torch::Tensor output;
  torch::Tensor block_table;
  torch::Tensor seq_lens;
  torch::Tensor v_cache;
  torch::Tensor output_lse;
  std::optional<torch::Tensor> q_quant_scale;
  std::optional<torch::Tensor> k_cache_quant_scale;
  std::optional<torch::Tensor> v_cache_quant_scale;
  std::optional<torch::Tensor> out_quant_scale;
  std::optional<torch::Tensor> alibi_slope;
  std::optional<torch::Tensor> mask;
  std::string compute_dtype;
  int max_seq_len;
  int window_size_left;
  int window_size_right;
  float scale;
  bool return_lse;
  int kv_cache_quant_bit_size;

  DecodeAttentionParams()
      : q_quant_scale(std::nullopt),
        k_cache_quant_scale(std::nullopt),
        v_cache_quant_scale(std::nullopt),
        out_quant_scale(std::nullopt),
        alibi_slope(std::nullopt),
        mask(std::nullopt),
        window_size_right(-1),
        return_lse(false),
        kv_cache_quant_bit_size(-1) {}
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
  bool store_output_before_norm;
  bool store_output_after_norm;
  bool dynamic_quant;

  FusedLayerNormParams()
      : residual(std::nullopt),
        beta(std::nullopt),
        bias(std::nullopt),
        quant_scale(std::nullopt),
        residual_out(std::nullopt),
        smooth_quant_scale(std::nullopt),
        normed_out(std::nullopt),
        store_output_before_norm(false),
        store_output_after_norm(false),
        dynamic_quant(false) {}
};

// Matmul parameters
struct MatmulParams {
  torch::Tensor a;
  torch::Tensor b;
  std::optional<torch::Tensor> bias;
  std::optional<torch::Tensor> c;
  double alpha;
  double beta;

  MatmulParams() : bias(std::nullopt), c(std::nullopt) {}
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
  int topk;
  bool renormalize;
  bool gated;
  std::string act_mode;
  std::string scoring_func;
  int start_expert_id;
  int block_n;
  bool avg_moe;
  std::optional<torch::Tensor> class_reduce_weight;
  std::optional<torch::Tensor> class_expert_id;
  std::optional<std::vector<bool>> w1_quant_flag;
  std::optional<std::vector<bool>> w2_quant_flag;
  int world_size;
  int shared_expert_num;
  std::string parallel_mode;

  FusedMoEParams()
      : bias1(std::nullopt),
        bias2(std::nullopt),
        residual(std::nullopt),
        input_smooth(std::nullopt),
        act_smooth(std::nullopt),
        w1_scale(std::nullopt),
        w2_scale(std::nullopt),
        w1_quant_flag(std::nullopt),
        w2_quant_flag(std::nullopt),
        scoring_func("softmax"),
        start_expert_id(0),
        block_n(0),
        avg_moe(false),
        class_reduce_weight(std::nullopt),
        class_expert_id(std::nullopt),
        w1_quant_flag(std::nullopt),
        w2_quant_flag(std::nullopt),
        world_size(0),
        shared_expert_num(0),
        parallel_mode("ep") {}
};
}  // namespace kernel
}  // namespace xllm