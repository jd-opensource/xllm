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
#include <tuple>

#include "attention.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"
#include "layers/mlu/deepseek_v32_sp_context.h"

namespace xllm {
namespace layer {

// Context structure for passing runtime arguments to kernel
struct IndexerRuntimeContext {
  // kernel input tensors
  torch::Tensor q;
  torch::Tensor weights;
  torch::Tensor k_cache_tensor;  // points to k (dense), k_full (gathered), or
                                 // k_cache (paged)
  std::optional<torch::Tensor> k_block_table;
  // length information
  std::optional<torch::Tensor> cu_seq_q_lens;
  torch::Tensor cu_seq_k_lens;
  torch::Tensor k_context_lens;
  // output buffers
  torch::Tensor new_block_tables;
  torch::Tensor new_context_lens;
  // temporary tensors ownership
  torch::Tensor _storage_k_full;
};

struct IndexerSPPreOut {
  torch::Tensor q;
  torch::Tensor k_local;
  torch::Tensor weights;
};

class IndexerImpl : public torch::nn::Module {
 public:
  IndexerImpl() = default;

  IndexerImpl(int64_t dim,
              int64_t index_n_heads,
              int64_t index_head_dim,
              int64_t qk_rope_head_dim,
              int64_t index_topk,
              int64_t q_lora_rank,
              bool enable_fused_qk,
              const std::shared_ptr<RotaryEmbeddingBase>& rotary_emb,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& qr,
      const torch::Tensor& positions,
      torch::Tensor& k_cache,
      const AttentionMetadata& attn_metadata,
      bool is_prefill,
      const std::optional<torch::Tensor>& mask = std::nullopt);

  IndexerSPPreOut sp_pre(const torch::Tensor& x,
                         const torch::Tensor& qr,
                         const torch::Tensor& positions,
                         const AttentionMetadata& attn_metadata,
                         const v32_sp::DeepseekV32SPContext& sp_ctx);

  v32_sp::PaddedGatherHandle sp_comm(
      const torch::Tensor& k_local,
      const v32_sp::DeepseekV32SPContext& sp_ctx);

  torch::Tensor sp_wait_k(const torch::Tensor& k_local,
                          const v32_sp::PaddedGatherHandle& gather_handle,
                          const v32_sp::DeepseekV32SPContext& sp_ctx);

  std::tuple<torch::Tensor, torch::Tensor> sp_post(
      const IndexerSPPreOut& pre_out,
      const torch::Tensor& k_global,
      torch::Tensor& k_cache,
      const AttentionMetadata& attn_metadata,
      const v32_sp::DeepseekV32SPMetadata& sp_meta,
      const v32_sp::DeepseekV32SPContext& sp_ctx);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t n_heads_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  int64_t index_topk_;
  float softmax_scale_;
  bool enable_fused_qk_;
  bool q_rope_at_front_;
  // Linear layers using ReplicatedLinear
  ReplicatedLinear wq_b_{nullptr};
  ReplicatedLinear wk_{nullptr};
  ReplicatedLinear weights_proj_{nullptr};
  RMSNorm k_norm_{nullptr};

  // Rotary embedding
  std::shared_ptr<RotaryEmbeddingBase> rotary_emb_;

  // Hadamard matrix
  torch::Tensor hadamard_matrix_;

  // Helper function for rotation activation
  torch::Tensor rotate_activation(const torch::Tensor& input,
                                  const torch::Tensor& hadamard_matrix);

  // Prepare runtime context for kernel
  IndexerRuntimeContext prepare_runtime_context(
      const torch::Tensor& k_current_dense,
      torch::Tensor& k_cache_paged,
      torch::Tensor& q,
      torch::Tensor& weights,
      const AttentionMetadata& attn_metadata,
      bool is_prefill,
      int64_t num_tokens);

  torch::Tensor preprocess_indexer_q(const torch::Tensor& qr,
                                     const torch::Tensor& positions,
                                     const AttentionMetadata& attn_metadata);

  std::tuple<torch::Tensor, torch::Tensor> preprocess_indexer_k(
      const torch::Tensor& x,
      const torch::Tensor& positions,
      torch::Tensor& k_cache,
      const AttentionMetadata& attn_metadata,
      bool write_k_cache);

  torch::Tensor preprocess_indexer_q_fused(const torch::Tensor& qr,
                                           const torch::Tensor& positions);

  torch::Tensor preprocess_indexer_k_fused(
      const torch::Tensor& x,
      const torch::Tensor& positions,
      torch::Tensor& k_cache,
      const AttentionMetadata& attn_metadata);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  preprocess_indexer_inputs(const torch::Tensor& x,
                            const torch::Tensor& qr,
                            const torch::Tensor& positions,
                            torch::Tensor& k_cache,
                            const AttentionMetadata& attn_metadata,
                            bool is_prefill,
                            bool write_k_cache = true);

  void write_prefill_k_cache(const torch::Tensor& k,
                             torch::Tensor& k_cache,
                             const torch::Tensor& slot_mapping);

  std::tuple<torch::Tensor, torch::Tensor> run_indexer_select_kernel(
      const AttentionMetadata& attn_metadata,
      bool is_prefill,
      IndexerRuntimeContext& ctx,
      const v32_sp::DeepseekV32SPMetadata* sp_meta = nullptr);
};

TORCH_MODULE(Indexer);

}  // namespace layer
}  // namespace xllm
