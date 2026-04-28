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

#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

// DSA cache type enum for DeepSeek V4 multi-cache management
enum class DSACacheType : int32_t {
  TOKEN = 0,           // block allocated by token count / ratio
  SEQUENCE = 1,        // one block per sequence
  SLIDING_WINDOW = 2,  // sliding window, fixed number of blocks per seq
};

// Per-cache metadata within a layer
struct DSACacheInfo {
  int32_t group_id;    // which block manager group this cache belongs to
  DSACacheType type;   // cache type
  int32_t ratio;       // compression ratio
  int32_t block_size;  // block size for this cache
};

// Group-level info
struct DSAGroupInfo {
  DSACacheType type;
  int32_t ratio;
  int32_t block_size;
};

namespace layer {

// DSAMetadata contains DeepSeek V4 sparse attention specific metadata,
// aligned with Python DSAMetadata(AttentionMetadata) class.
// It is built once at the beginning of model forward pass and reused by
// all layers. Use DSAMetadataBuilder to build instances from ModelInputParams.
struct DSAMetadata {
  // ===== Fields from Python AttentionMetadata base class =====
  // seq_lens: kv sequence lengths (context_length)
  torch::Tensor seq_lens;
  // seq_lens_q: query sequence lengths
  torch::Tensor seq_lens_q;
  // attn_mask: attention mask
  torch::Tensor attn_mask;
  // cos_table / sin_table: base RoPE cos/sin tables
  torch::Tensor cos_table;
  torch::Tensor sin_table;

  // ===== DSA-specific fields =====
  // layer_id: current layer (Python per-layer, C++ shared across layers)
  int32_t layer_id = 0;
  // num_speculative_tokens: number of speculative decoding tokens
  int32_t num_speculative_tokens = 0;

  // cp_input_dict: context-parallel inputs placeholder (reserved, optional)
  std::unordered_map<std::string, torch::Tensor> cp_input_dict;

  // RoPE caches selected for the current layer's q/kv/output RoPE.
  torch::Tensor cos;
  torch::Tensor sin;
  // RoPE caches for compressor/indexer paths, indexed by compressed positions.
  torch::Tensor c4_cos;
  torch::Tensor c4_sin;
  torch::Tensor c128_cos;
  torch::Tensor c128_sin;
  torch::Tensor start_pos;

  // Multi-manager block tables and slot mappings
  // Indexed as [layer_id][cache_idx] after expansion by build_forward_context.
  // Same-group caches share the same underlying tensor (no copy).
  std::vector<std::vector<torch::Tensor>> block_tables;
  std::vector<std::vector<torch::Tensor>> slot_mappings;

  // Sequence length metadata
  // actual_seq_lengths_kv: (batch_size,) — per-seq kv context length
  torch::Tensor actual_seq_lengths_kv;
  // actual_seq_lengths_query: (batch_size+1,) — cumsum of per-seq query lengths
  //   prefill: pad(cumsum(context_length), (1,0), 0)
  //   decode:  pad(cumsum(ones(batch_size)), (1,0), 0)
  torch::Tensor actual_seq_lengths_query;
  // max_seqlen_kv / max_seqlen_q: max sequence lengths
  torch::Tensor max_seqlen_kv;
  torch::Tensor max_seqlen_q;

  // Compressed positions
  // input_positions: (total_tokens,) — token position IDs
  torch::Tensor input_positions;
  // c4_pad_positions: positions for C4 compressed RoPE
  torch::Tensor c4_pad_positions;
  // c128_pad_positions: positions for C128 compressed RoPE
  torch::Tensor c128_pad_positions;

  // Precomputed sparse/indexer metadata tensors (Python forward aligned).
  // Built once per model forward before layer iteration.
  torch::Tensor c1_metadata;
  torch::Tensor c4_metadata;
  torch::Tensor c128_metadata;
  torch::Tensor qli_metadata;

  // hadamard: Hadamard transform matrix
  torch::Tensor hadamard;

  // Cache spec per layer
  // caches_info[layer_id][cache_idx] = {group_id, type, ratio, block_size}
  // Points to model-owned data; valid for the lifetime of the model.
  const std::vector<std::vector<DSACacheInfo>>* caches_info = nullptr;
};

}  // namespace layer
}  // namespace xllm
