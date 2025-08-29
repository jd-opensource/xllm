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

#include "qwen3_attention.h"

#include <glog/logging.h>

#include <tuple>

#include "core/layers/rotary_embedding.h"

namespace xllm {

torch::Tensor _compute_inv_freq(float base, int64_t rotary_dim) {
  // 1 / (base ** (torch.arange(0, rotary_dim, 2, float) / rotary_dim))
  auto t = torch::arange(/*start=*/0,
                         /*end=*/rotary_dim,
                         /*step=*/2,
                         torch::TensorOptions().dtype(torch::kFloat));
  return 1.0 / torch::pow(base, t / static_cast<double>(rotary_dim));
}

Qwen3Attention::Qwen3Attention(const ModelArgs& args,
                               const QuantArgs& quant_args,
                               const ParallelArgs& parallel_args,
                               const torch::TensorOptions& options) {
  const int64_t tp_size = parallel_args.world_size();
  const int64_t total_num_heads = args.num_attention_heads();
  const int64_t total_num_kv_heads =
      args.n_kv_heads().value_or(args.num_attention_heads());

  CHECK(total_num_heads % tp_size == 0);
  num_heads_ = total_num_heads / tp_size;

  if (total_num_kv_heads >= tp_size) {
    CHECK(total_num_kv_heads % tp_size == 0);
    num_kv_heads_ = total_num_kv_heads / tp_size;
  } else {
    CHECK(tp_size % total_num_kv_heads == 0);
    num_kv_heads_ = 1;
  }

  head_dim_ = args.hidden_size() / total_num_heads;
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  scaling_ = std::sqrt(1.0f / head_dim_);

  // 1. QKV parallel linear
  qkv_proj_ = register_module(
      "qkv_proj",
      ColumnParallelLinear(args.hidden_size(),
                           (num_heads_ + 2 * num_kv_heads_) * head_dim_,
                           /*bias=*/true,
                           /*gather_output=*/false,
                           quant_args,
                           parallel_args,
                           options));

  // 2. Output projection
  o_proj_ = register_module("o_proj",
                            RowParallelLinear(num_heads_ * head_dim_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args,
                                              options));

  // 3. RMSNorm
  q_norm_ = register_module("q_norm",
                            RMSNorm(head_dim_, args.rms_norm_eps(), options));

  k_norm_ = register_module("k_norm",
                            RMSNorm(head_dim_, args.rms_norm_eps(), options));

  // 4. Rotary embedding
  torch::Tensor inv_freq = _compute_inv_freq(args.rope_theta(), head_dim_);
  float sm_scale = scaling_;
  bool interleaved = false;
  rotary_emb_ = create_rotary_embedding(args,
                                        /*rotary_dim=*/head_dim_,
                                        inv_freq,
                                        interleaved,
                                        sm_scale,
                                        options);

  // 5. Attention
  attn_ = xllm::mlu::Attention(
      num_heads_, head_dim_, scaling_, num_kv_heads_, args.sliding_window());
}

torch::Tensor Qwen3Attention::forward(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const xllm::mlu::AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);

  auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
  auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + 2 * kv_size_);

  const int64_t T = q.size(0);

  // 2. q-norm
  q = q.view({T, num_heads_, head_dim_});
  q = q_norm_->forward(q);
  q = q.view({T, q_size_});

  // 3. k-norm
  k = k.view({T, num_kv_heads_, head_dim_});
  k = k_norm_->forward(k);
  k = k.view({T, kv_size_});

  // 4. rope
  std::tie(q, k) = rotary_emb_->forward(q.view({T, num_heads_, head_dim_}),
                                        k.view({T, num_kv_heads_, head_dim_}),
                                        positions);

  q = q.view({T, num_heads_, head_dim_});
  k = k.view({T, num_kv_heads_, head_dim_});
  v = v.view({T, num_kv_heads_, head_dim_});

  // 5. store k/v cache and do attention
  auto out = std::get<0>(attn_.forward(attn_metadata, q, k, v, kv_cache));

  // 6. output projection
  return o_proj_->forward(out);
}

void Qwen3Attention::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict.get_dict_with_prefix("qkv_proj."));
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("o_proj."));
}

}  // namespace xllm
