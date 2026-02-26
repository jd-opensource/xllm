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

#include "qwen3_next_attention.h"
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <glog/logging.h>
#include <tuple>
namespace xllm {
namespace layer {

Qwen3NextAttentionImpl::Qwen3NextAttentionImpl(const ModelArgs& args,
                                               const QuantArgs& quant_args,
                                               const ParallelArgs& parallel_args,
                                               const torch::TensorOptions& options,
                                               int32_t layer_id) {
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  const int64_t total_num_heads = args.n_heads();
  const int64_t total_num_kv_heads = args.n_kv_heads().value_or(args.n_heads());
  layer_id_ = layer_id;
  rank_ = parallel_args.tp_group_->rank();
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
  scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  attn_output_gate_ = args.attn_output_gate();
  // 1. QKV linear
  qkv_proj_ = register_module("qkv_proj",
                              QKVParallelLinear(args.hidden_size(),
                                                attn_output_gate_ ? num_heads_ * 2 : num_heads_,  
                                                num_kv_heads_,
                                                args.head_dim(),
                                                num_kv_head_replicas_,
                                                /*bias=*/args.attention_bias(),
                                                /*gather_output=*/false,
                                                parallel_args,
                                                options));

  // 2. O proj
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(total_num_heads * head_dim_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args,
                                              options));

  // 3. Q norm
  q_norm_ = register_module("q_norm",
                            Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));

  // 4. K norm
  k_norm_ = register_module("k_norm",
                            Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));
  
  // 5. Rotary embedding
  const int rotary_dim = static_cast<int>(head_dim_ * args.partial_rotary_factor());
  rotary_emb_ = register_module(
    "rotary_emb",
    PartialRotaryEmbedding(rotary_dim,
                           args.max_position_embeddings(),
                           args.rope_theta(),
                           head_dim_,
                           true,
                           false,
                           options));

  // 6. Attention
  attn_ = register_module("attn",
                          Attention(num_heads_,
                                    head_dim_,
                                    scaling_,
                                    num_kv_heads_,
                                    args.sliding_window()));
}

torch::Tensor Qwen3NextAttentionImpl::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);
  torch::Tensor q, k, v;
  torch::Tensor gate;
  
  if (attn_output_gate_) {
    // Split qkv for attn_output_gate case: [q_size*2, kv_size, kv_size]
    auto q_gate = qkv.slice(/*dim=*/-1, 0, q_size_ * 2);
    k = qkv.slice(/*dim=*/-1, q_size_ * 2, q_size_ * 2 + kv_size_);
    v = qkv.slice(/*dim=*/-1, q_size_ * 2 + kv_size_, q_size_ * 2 + kv_size_ * 2);
    v = v.contiguous();

    std::vector<int64_t> orig_shape;
    int64_t q_gate_dim = q_gate.dim();
    orig_shape = std::vector<int64_t>(
        q_gate.sizes().slice(0, q_gate_dim - 1).begin(),
        q_gate.sizes().slice(0, q_gate_dim - 1).end()
    );

    std::vector<int64_t> new_shape = orig_shape;
    new_shape.push_back(num_heads_);
    int64_t orig_total = 1;
    for (auto d : orig_shape) orig_total *= d;
    int64_t last_dim = q_gate.numel() / (orig_total * num_heads_);
    new_shape.push_back(last_dim);

    torch::Tensor q_gate_reshaped = q_gate.reshape(new_shape);

    auto chunks = torch::chunk(q_gate_reshaped, 2, /*dim=*/-1);
    q = chunks[0];
    gate = chunks[1];

    std::vector<int64_t> q_new_shape = orig_shape;
    q_new_shape.push_back(q.numel() / orig_total);
    q = q.reshape(q_new_shape);

    std::vector<int64_t> gate_new_shape = orig_shape;
    gate_new_shape.push_back(gate.numel() / orig_total);
    gate = gate.reshape(gate_new_shape);
  } else {
    // Normal case: [q_size, kv_size, kv_size]
    q = qkv.slice(/*dim=*/-1, 0, q_size_);
    k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
    v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);
  }

  const int64_t T = q.size(0);

  auto q_reshaped = q.reshape({T, num_heads_, head_dim_});
  auto q_normed = q_norm_->forward(q_reshaped);
  auto k_reshaped = k.reshape({T, num_kv_heads_, head_dim_});
  auto k_normed = k_norm_->forward(k_reshaped);

  q = q_normed.view({T, q_size_});
  k = k_normed.view({T, kv_size_});

  rotary_emb_->forward(positions,q,k);
  auto out = std::get<0>(attn_->forward(attn_metadata, q, k, v, kv_cache));
  
  if (attn_output_gate_) {
    gate = torch::sigmoid(gate);
    out = out * gate;
  }

  out = o_proj_->forward(out);
  return out;
}

void Qwen3NextAttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict);
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
  if (auto w = state_dict.get_tensor("q_norm.weight"); w.defined()) {
    q_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  if (auto w = state_dict.get_tensor("k_norm.weight"); w.defined()) {
    k_norm_->load_state_dict(StateDict({{"weight", w}}));
  }
}

}  // namespace layer
}  // namespace xllm
