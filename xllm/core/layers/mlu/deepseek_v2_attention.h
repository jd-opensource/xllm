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

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/indexer.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"

namespace xllm {
namespace layer {

class DeepseekV2AttentionImpl : public torch::nn::Module {
 public:
  DeepseekV2AttentionImpl() = default;
  DeepseekV2AttentionImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache);

  void load_state_dict(const StateDict& state_dict);

 private:
  bool is_per_token_smoothquant_ = false;
  bool use_fused_mla_qkv_ = false;
  bool enable_lighting_indexer_ = false;
  bool has_trans_ = false;
  int64_t num_local_heads_;
  int64_t qk_head_dim_;
  int64_t v_head_dim_;
  int64_t q_lora_rank_;
  int64_t kv_lora_rank_;
  int64_t qk_nope_head_dim_;
  int64_t qk_rope_head_dim_;
  int64_t index_topk_;
  torch::Tensor w_kc_;
  torch::Tensor w_vc_;
  torch::Tensor weight_c_;

  ReplicatedLinear q_a_proj_{nullptr};
  ColumnParallelLinear q_b_proj_{nullptr};
  ColumnParallelLinear q_proj_{nullptr};
  RMSNorm q_a_layernorm_{nullptr};

  ReplicatedLinear kv_a_proj_with_mqa_{nullptr};
  RMSNorm kv_a_layernorm_{nullptr};

  ColumnParallelLinear kv_b_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};

  Attention attn_{nullptr};
  DeepseekScalingRotaryEmbedding rotary_emb_{nullptr};
  Indexer indexer_{nullptr};
};
TORCH_MODULE(DeepseekV2Attention);

}  // namespace layer
}  // namespace xllm
