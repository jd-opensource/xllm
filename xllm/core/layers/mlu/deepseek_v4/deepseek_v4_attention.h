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

#include <optional>
#include <tuple>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"
#include "layers/mlu/deepseek_v4/compressor.h"
#include "layers/mlu/deepseek_v4/indexer_v2.h"

namespace xllm {
namespace layer {
// Generate window top-k indices for sparse attention
// For decode: returns indices within sliding window
// For prefill: returns causal window mask per query position
// Args:
//   window_size: Sliding window size
//   q_cu_seq_lens: Cumulative query lengths [batch_size + 1]
//   seq_lens: Current sequence lengths [batch_size]
//   device: Device for output tensors
// Returns:
//   Vector of tensors, each [seq_len, K] with -1 for invalid indices
std::vector<torch::Tensor> get_window_topk_idxs(
    int64_t window_size,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& seq_lens,
    torch::Device device);

// Generate compressed top-k indices for sparse attention
// Maps query positions to their valid compressed KV positions
// Args:
//   compress_ratio: Compression ratio
//   q_cu_seq_lens: Cumulative query lengths [batch_size + 1]
//   seq_lens: Current sequence lengths [batch_size]
//   offsets: Offset for compressed indices per batch [batch_size]
//   device: Device for output tensors
// Returns:
//   Vector of tensors, each [seq_len, K] with -1 for invalid indices
std::vector<torch::Tensor> get_compress_topk_idxs(
    int64_t compress_ratio,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& seq_lens,
    const torch::Tensor& offsets,
    torch::Device device);

// DeepSeek V4 attention implementation with MLA (Multi-head Latent Attention)
// Features:
// - LoRA-based query projection (wq_a -> q_norm -> wq_b)
// - Shared KV projection with RoPE
// - Sparse attention with sliding window + compressed KV
// - Optional indexer for dynamic top-k selection
// - Grouped output projection (wo_a -> wo_b)
class DeepSeekV4AttentionImpl : public torch::nn::Module {
 public:
  DeepSeekV4AttentionImpl() = default;

  // Constructor
  // Args:
  //   args: Model configuration arguments
  //   quant_args: Quantization configuration
  //   parallel_args: Tensor parallelism configuration
  //   options: Tensor options (dtype, device)
  //   layer_id: Layer index for per-layer configuration
  //   cached_state_num: Number of cached states for compressor/indexer
  DeepSeekV4AttentionImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options,
                          int64_t layer_id,
                          int64_t cached_state_num);

  // Forward pass
  // Args:
  //   positions: Token positions [num_tokens]
  //   hidden_states: Input hidden states [num_tokens, hidden_size]
  //   attn_metadata: Attention metadata (seq_lens, block_tables, etc.)
  //   kv_cache: KV cache for attention
  //   batch_to_kv_state: Mapping from batch index to KV state index
  // Returns:
  //   Output tensor [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& positions,
                        const torch::Tensor& hidden_states,
                        const AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const std::vector<int64_t>& batch_to_kv_state);

  void load_state_dict(const StateDict& state_dict);

 private:
  // Core sparse attention forward with window + compressed KV
  torch::Tensor forward_sparse_attn(
      const torch::Tensor& positions,
      const torch::Tensor& hidden_states,
      torch::Tensor& k_cache,
      torch::Tensor& indexer_cache,
      const AttentionMetadata& attn_metadata,
      const std::vector<int64_t>& batch_to_kv_state);

  // Convert top-k indices to block tables for paged attention
  // Args:
  //   topk_idxs_list: List of top-k indices per batch [seq_len, K]
  //   window_size: Sliding window size
  //   max_model_len: Maximum model length
  //   compress_ratio: Compression ratio
  //   offsets: Offset per batch for prefill mode
  //   block_tables: Original block tables for decode mode
  //   is_prefill: Whether in prefill mode
  // Returns:
  //   (new_block_tables, new_context_lens) for sparse attention
  std::tuple<torch::Tensor, torch::Tensor> convert_topk_to_block_tables(
      const std::vector<torch::Tensor>& topk_idxs_list,
      int64_t window_size,
      int64_t max_model_len,
      int64_t compress_ratio,
      const std::optional<torch::Tensor>& offsets,
      const torch::Tensor& block_tables,
      bool is_prefill);

  // Write KV to paged cache with window wrapping
  void write_kv_to_cache(const torch::Tensor& kv,
                         torch::Tensor& k_cache,
                         const torch::Tensor& block_table,
                         const torch::Tensor& q_cu_seq_lens,
                         const torch::Tensor& seq_lens,
                         bool is_prefill);

 private:
  // Model dimensions
  int64_t layer_id_;
  int64_t hidden_size_;
  int64_t num_heads_;
  int64_t num_local_heads_;
  int64_t head_dim_;
  int64_t q_lora_rank_;
  int64_t rope_head_dim_;
  int64_t o_lora_rank_;
  int64_t o_groups_;
  int64_t o_local_groups_;
  int64_t window_size_;
  int64_t compress_ratio_;
  int64_t max_model_len_;
  int64_t original_seq_len_;

  // Scaling and normalization
  float softmax_scale_;
  double eps_;

  // Tensor parallel
  int64_t tp_size_;
  int64_t tp_rank_;
  ProcessGroup* tp_group_;

  // Query projection with LoRA
  ReplicatedLinear wq_a_{nullptr};  // [hidden_size, q_lora_rank]
  RMSNorm q_norm_{nullptr};
  ColumnParallelLinear wq_b_{nullptr};  // [q_lora_rank, num_heads * head_dim]

  // KV projection
  ReplicatedLinear wkv_{nullptr};  // [hidden_size, head_dim]
  RMSNorm kv_norm_{nullptr};

  // Output projection
  // Two variants based on TP size vs o_groups
  ColumnParallelLinear wo_a_col_{nullptr};
  RowParallelLinear wo_b_row_{nullptr};
  ReplicatedLinear wo_a_rep_{nullptr};
  ReplicatedLinear wo_b_rep_{nullptr};
  bool use_parallel_o_proj_;

  // Attention sink (learnable bias for softmax normalization)
  DEFINE_WEIGHT(attn_sink);

  // Compressor for KV state compression
  Compressor compressor_{nullptr};

  // Indexer for dynamic top-k selection (compress_ratio == 4)
  IndexerV2 indexer_{nullptr};

  // Precomputed frequency tensor for RoPE
  torch::Tensor freqs_cis_;

  // Rotary embedding
  DeepseekScalingRotaryEmbedding rotary_emb_{nullptr};
  DeepseekScalingRotaryEmbedding output_rotary_emb_{nullptr};
};
TORCH_MODULE(DeepSeekV4Attention);

}  // namespace layer
}  // namespace xllm
