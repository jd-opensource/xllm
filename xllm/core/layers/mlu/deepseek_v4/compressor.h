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
#include <tuple>
#include <vector>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "layers/common/linear.h"
#include "layers/common/rms_norm.h"
#include "layers/common/rotary_embedding.h"

namespace xllm {
namespace layer {

class CompressorImpl : public torch::nn::Module {
 public:
  CompressorImpl(int64_t dim,
                 int64_t head_dim,
                 int64_t rope_head_dim,
                 int64_t compress_ratio,
                 int64_t cached_state_num,
                 double norm_eps,
                 bool rotate,
                 DeepseekScalingRotaryEmbedding& rotary_emb,
                 const torch::TensorOptions& options);

  // Compresses KV states using gated attention mechanism
  // Returns: (compressed_kvs, compress_lens) for each batch
  std::tuple<std::vector<torch::Tensor>, std::vector<int64_t>> forward(
      const torch::Tensor& x,
      const torch::Tensor& positions,
      const torch::Tensor& block_tables,
      const torch::Tensor& q_cu_seq_lens,
      const torch::Tensor& seq_lens,
      const std::vector<int64_t>& batch_to_kv_state,
      torch::Tensor& kv_cache,
      int64_t window_offset,
      const torch::Tensor& freqs_cis);

  void load_state_dict(const StateDict& state_dict);

 private:
  // Dimension constants
  int64_t dim_;
  int64_t head_dim_;
  int64_t rope_head_dim_;
  int64_t nope_head_dim_;
  int64_t compress_ratio_;  // Compression ratio (e.g., 4 or 8)

  // Configuration flags (declared before coeff_ since coeff_ depends on
  // overlap_)
  bool overlap_;    // Enabled when compress_ratio == 4
  int64_t coeff_;   // Compression coefficient: 1 + overlap
  bool rotate_;     // Apply Hadamard rotation
  bool converted_;  // Tracks if ape has been reordered for overlap mode
  double norm_eps_;

  // Learnable parameters
  DEFINE_WEIGHT(ape);  // Absolute position embedding for scoring

  // Sub-modules
  ReplicatedLinear wkv_{nullptr};    // Projects input to KV states
  ReplicatedLinear wgate_{nullptr};  // Projects input to gate scores
  RMSNorm norm_{nullptr};
  DeepseekScalingRotaryEmbedding rotary_emb_{nullptr};

  // State buffers: [cached_state_num, state_ratio, state_dim]
  torch::Tensor kv_state_;     // Cached KV states for incremental compression
  torch::Tensor score_state_;  // Cached gate scores for weighting
  torch::Tensor hadamard_matrix_;  // For rotation if enabled

  // Reorders ape for overlap mode (first time only)
  void convert_ape_if_needed();

  // Updates state buffers with remainder tokens
  void update_state_indices(int64_t state_idx,
                            int64_t offset,
                            const torch::Tensor& kv_remainder,
                            const torch::Tensor& score_remainder,
                            int64_t remainder);

  // Compresses overlapping states using softmax-weighted sum
  torch::Tensor compress_with_overlap(int64_t state_idx);

  // Transforms tensor for overlap mode (shifts and concatenates)
  torch::Tensor overlap_transform(const torch::Tensor& tensor,
                                  float fill_value);

  // Applies RoPE to compressed KV using cutoff_positions
  void apply_rope_to_compressed_kv(torch::Tensor& kv,
                                   const torch::Tensor& cutoff_positions);
};

TORCH_MODULE(Compressor);

// Apply rotary position embedding by treating last dim as complex numbers
void apply_rotary_emb(torch::Tensor& x,
                      const torch::Tensor& freqs_cis,
                      bool inverse = false);

}  // namespace layer
}  // namespace xllm