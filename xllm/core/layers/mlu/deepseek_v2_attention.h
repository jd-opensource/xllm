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

#include <memory>
#include <optional>

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"
#include "layers/mlu/deepseek_v32_sp_context.h"
#include "layers/mlu/indexer.h"
#include "platform/stream.h"

namespace xllm {
namespace layer {

class DeepseekV2AttentionImpl : public torch::nn::Module {
 public:
  DeepseekV2AttentionImpl() = default;
  DeepseekV2AttentionImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options,
                          const OptimizationConfig& optimization_config);

  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const v32_sp::DeepseekV32SPContext* sp_ctx = nullptr);

  // whether to use full linear weights for projection
  bool use_full_linear_weights() const {
    return use_full_replicated_attention_weights_;
  }

  bool can_use_sp() const {
    return enable_lighting_indexer_ && use_full_linear_weights() &&
           static_cast<bool>(attn_full_);
  }

  int64_t local_compute_head_count() const { return num_local_heads_; }

  int64_t local_compute_projection_width() const {
    return num_local_heads_ * v_head_dim_;
  }

  void load_state_dict(const StateDict& state_dict);

 private:
  struct SPQRPreOut {
    torch::Tensor q;
    torch::Tensor qr;
  };

  struct SPMLAPreOut {
    torch::Tensor qr;
    torch::Tensor q_input;
    torch::Tensor k_input;
    torch::Tensor v_input;
  };

  torch::Tensor forward_normal_tp(const torch::Tensor& positions,
                                  const torch::Tensor& hidden_states,
                                  const AttentionMetadata& attn_metadata,
                                  KVCache& kv_cache,
                                  bool is_prefill_or_chunked_prefill);

  // ===== sequence parallel related =====
  torch::Tensor forward_sp(const torch::Tensor& positions,
                           const torch::Tensor& hidden_states,
                           const AttentionMetadata& attn_metadata,
                           const v32_sp::DeepseekV32SPContext& sp_ctx,
                           KVCache& kv_cache,
                           bool is_prefill_or_chunked_prefill);
  SPQRPreOut sp_qr_pre(const torch::Tensor& hidden_states);
  SPMLAPreOut sp_mla_pre(const torch::Tensor& hidden_states,
                         const torch::Tensor& positions,
                         const SPQRPreOut& qr_pre,
                         const v32_sp::DeepseekV32SPContext& sp_ctx,
                         const torch::TensorOptions& options);
  v32_sp::PaddedGatherHandle sp_mla_comm(
      const torch::Tensor& k_input,
      const v32_sp::DeepseekV32SPContext& sp_ctx) const;
  void sp_mla_finish_k(SPMLAPreOut& pre_out,
                       const v32_sp::PaddedGatherHandle& k_handle,
                       const v32_sp::DeepseekV32SPContext& sp_ctx) const;
  torch::Tensor project_sp_output(const torch::Tensor& attn_output);

  void decode_q_pre_base(torch::Tensor& q,
                         torch::Tensor& qr,
                         torch::Tensor& q_input,
                         const torch::Tensor& positions,
                         const AttentionMetadata& attn_metadata,
                         bool use_prompt_rope);
  void decode_kv_pre_base(torch::Tensor& latent_cache,
                          const torch::Tensor& positions,
                          const AttentionMetadata& attn_metadata,
                          bool use_prompt_rope);
  void decode_qkv_pre_fused(torch::Tensor& q,
                            torch::Tensor& qr,
                            torch::Tensor& q_input,
                            torch::Tensor& latent_cache,
                            torch::Tensor& kv_cache,
                            std::optional<torch::Tensor> k_cache_scale,
                            const torch::Tensor& positions,
                            const AttentionMetadata& attn_metadata,
                            bool use_prompt_rope);

  void prepare_mla_inputs(torch::Tensor& q,
                          torch::Tensor& qr,
                          torch::Tensor& q_input,
                          torch::Tensor& latent_cache,
                          const torch::Tensor& hidden_states,
                          torch::Tensor& k_cache,
                          std::optional<torch::Tensor> k_cache_scale,
                          const torch::Tensor& positions,
                          const AttentionMetadata& attn_metadata,
                          bool enable_fused_qkv,
                          bool use_prompt_rope);

  AttentionMetadata build_mla_attention_metadata(
      const torch::Tensor& positions,
      const torch::Tensor& hidden_states,
      const torch::Tensor& qr,
      const torch::Tensor& k_input,
      const AttentionMetadata& attn_metadata,
      KVCache& kv_cache,
      std::optional<torch::Tensor> k_cache_scale,
      bool is_prefill_phase,
      const std::optional<torch::Tensor>& new_block_tables = std::nullopt,
      const std::optional<torch::Tensor>& new_context_lens = std::nullopt);

  torch::Tensor project_mla_output(const torch::Tensor& attn_output,
                                   int64_t q_len);

 private:
  bool use_full_replicated_attention_weights_ = false;
  bool is_per_token_smoothquant_ = false;
  bool use_fused_mla_qkv_ = false;
  bool enable_lighting_indexer_ = false;
  bool has_trans_ = false;
  bool interleaved_ = false;
  double eps_;
  int64_t tp_rank_ = 0;
  int64_t tp_world_size_ = 1;
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
  Attention attn_full_{nullptr};
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;
  std::shared_ptr<RotaryEmbeddingBase> indexer_rotary_emb_;
  Indexer indexer_{nullptr};
  std::unique_ptr<Stream> sp_comm_stream_;
};
TORCH_MODULE(DeepseekV2Attention);

}  // namespace layer
}  // namespace xllm
