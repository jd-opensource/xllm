/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_attention.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_topk.h"
#include "layers/mlu/deepseek_v4/hyper_connection.h"
#include "layers/mlu/fused_moe.h"

namespace xllm {
namespace layer {

// DeepSeek V4 Decoder Layer implementation
// Features:
// - HyperConnection for dynamic residual connections (pre/post modules)
// - DeepSeekV4Attention with MLA and sparse attention
// - FusedMoE for mixture-of-experts feed-forward network
// - Only supports pure TP (Tensor Parallel) and EP (Expert Parallel) strategies
class DeepSeekV4DecoderLayerImpl : public torch::nn::Module {
 public:
  explicit DeepSeekV4DecoderLayerImpl(const ModelContext& context,
                                      int32_t layer_id,
                                      int64_t cached_state_num);

  ~DeepSeekV4DecoderLayerImpl() override = default;

  void load_state_dict(const StateDict& state_dict);

  // Forward pass
  // Args:
  //   x: Input hidden states [num_tokens, hc_mult, hidden_size]
  //   residual: Optional residual hidden states [num_tokens, hc_mult,
  //   hidden_size] positions: Token positions [num_tokens] attn_metadata:
  //   Attention metadata (seq_lens, block_tables, etc.) kv_cache: KV cache for
  //   attention input_params: Model input parameters batch_to_kv_state: Mapping
  //   from batch index to KV state index input_ids: Optional input token ids
  //   for hash-based MoE routing
  // Returns:
  //   Output tensor [num_tokens, hc_mult, hidden_size]
  torch::Tensor forward(
      torch::Tensor& x,
      std::optional<torch::Tensor>& residual,
      torch::Tensor& positions,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      const ModelInputParams& input_params,
      const std::vector<int64_t>& batch_to_kv_state,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

 private:
  // Layer configuration
  int32_t layer_id_;
  int64_t hidden_size_;
  float norm_eps_;

  // HyperConnection parameters
  int64_t hc_mult_;
  int64_t hc_sinkhorn_iters_;
  float hc_eps_;

  // Parallel configuration
  ParallelArgs parallel_args_;

  // HyperConnection modules for attention
  HyperConnectionPre hc_attn_pre_{nullptr};
  HyperConnectionPost hc_attn_post_{nullptr};

  // HyperConnection modules for MoE
  HyperConnectionPre hc_moe_pre_{nullptr};
  HyperConnectionPost hc_moe_post_{nullptr};

  // Attention layer
  DeepSeekV4Attention attention_{nullptr};

  // Normalization layers
  RMSNorm attn_norm_{nullptr};
  RMSNorm moe_norm_{nullptr};

  // MoE MLP layer
  bool use_hash_;
  ReplicatedLinear route_gate_{nullptr};
  DeepSeekV4TopK topk_{nullptr};
  FusedMoE moe_mlp_{nullptr};
};

TORCH_MODULE(DeepSeekV4DecoderLayer);

}  // namespace layer
}  // namespace xllm
