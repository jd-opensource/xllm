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

#include "qwen2_attention.h"

#include <glog/logging.h>

#include <tuple>

namespace {
inline bool is_qwen3_model(const std::string& model_type) {
  static const std::set<std::string> qwen3_type_set = {
      "qwen3", "qwen3_vl", "qwen3_moe", "qwen3_vl_moe"};
  return qwen3_type_set.contains(model_type);
}
}  // namespace

namespace xllm {
namespace layer {

Qwen2AttentionImpl::Qwen2AttentionImpl(const ModelContext& context) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  is_qwen3_style_ = is_qwen3_model(args.model_type());
  bool qkv_bias = is_qwen3_style_ ? args.attention_bias() : true;

  CHECK(total_num_heads % tp_size == 0);
  num_heads_ = total_num_heads / tp_size;

  if (total_num_kv_heads >= tp_size) {
    CHECK(total_num_kv_heads % tp_size == 0);
    num_kv_heads_ = total_num_kv_heads / tp_size;
    num_kv_head_replicas_ = 1;
  } else {
    CHECK(tp_size % total_num_kv_heads == 0);
    num_kv_heads_ = 1;
    num_kv_head_replicas_ = tp_size / total_num_kv_heads;
  }

  head_dim_ = args.head_dim();
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  scaling_ = std::sqrt(1.0f / head_dim_);

  // 1. QKV parallel linear
  qkv_proj_ = register_module("qkv_proj",
                              QKVParallelLinear(args.hidden_size(),
                                                num_heads_,
                                                num_kv_heads_,
                                                args.head_dim(),
                                                num_kv_head_replicas_,
                                                qkv_bias,
                                                /*gather_output=*/false,
                                                parallel_args,
                                                options));

  // 2. Output projection
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * args.head_dim(),
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/true,
                                              quant_args,
                                              parallel_args,
                                              options));

  // 3. RMSNorm
  if (is_qwen3_style_) {
    q_norm_ = register_module(
        "q_norm", RMSNorm(args.head_dim(), args.rms_norm_eps(), options));

    k_norm_ = register_module(
        "k_norm", RMSNorm(args.head_dim(), args.rms_norm_eps(), options));
  }

  // 4. Rotary embedding
  rotary_emb_ =
      register_module("rope",
                      MRotaryEmbedding(/*rotary_dim=*/head_dim_,
                                       args.max_position_embeddings(),
                                       args.rope_theta(),
                                       /*interleaved=*/false,
                                       args.rope_scaling_mrope_section(),
                                       options));

  // 5. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));
}

torch::Tensor Qwen2AttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);

  auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
  auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

  const int64_t T = q.size(0);

  if (is_qwen3_style_) {
    // 2. q-norm
    q = std::get<0>(q_norm_->forward(q));

    // 3. k-norm
    k = std::get<0>(k_norm_->forward(k));
  }

  // 4. rope
  rotary_emb_->forward(q, k, positions, attn_metadata);
  q = q.view({T, q_size_});
  k = k.view({T, kv_size_});

  // 5. store k/v cache and do attention
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));

  // 6. output projection
  return o_proj_->forward(out);
}

void Qwen2AttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict, {"q_proj.", "k_proj.", "v_proj."});
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  if (is_qwen3_style_) {
    q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
    k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
  }
}

}  // namespace layer
}  // namespace xllm
