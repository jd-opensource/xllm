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

#include <tuple>
#include <utility>
#include <vector>

namespace xllm::kernel::npu::tilelang {

// Public TileLang kernel APIs exported to the xLLM NPU runtime.
//
// Apply TileLang RoPE kernel in-place on a single input tensor.
// Invalid inputs trigger CHECK failures.
// Supports input not contiguous, with stride.
void rope_in_place(torch::Tensor& input,
                   const torch::Tensor& sin_cache,
                   const torch::Tensor& cos_cache);

// Compute fused GDN gating outputs on NPU.
// Invalid inputs trigger CHECK failures.
std::pair<torch::Tensor, torch::Tensor> fused_gdn_gating(
    const torch::Tensor& A_log,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& dt_bias,
    float softplus_beta,
    float softplus_threshold);

// Fused ACL-graph metadata updater for decode-side persistent buffers.
//
// All mandatory metadata tensors are int32 NPU tensors. Optional groups are
// enabled by defining both src and dst tensors:
// - q_cu_seq_lens
// - linear_state_indices
// - input_embedding
//
// positions:
// - non-mRoPE: 1D [actual_num_tokens]
// - mRoPE: 2D [3, actual_num_tokens] for src and [3, capacity] for dst
//
// block_tables:
// - src: 2D [actual_batch_size, actual_block_table_len]
// - dst: 2D [capacity, dst_block_table_stride]
//
// input_embedding:
// - src: 2D [actual_num_tokens, hidden_size]
// - dst: 2D [capacity, hidden_size]
struct ModelInputBufferUpdaterParams {
  torch::Tensor src_tokens;
  torch::Tensor src_positions;
  torch::Tensor src_new_cache_slots;
  torch::Tensor src_q_seq_lens;
  torch::Tensor src_kv_seq_lens;
  torch::Tensor src_q_cu_seq_lens;
  torch::Tensor src_linear_state_indices;
  torch::Tensor src_block_tables;
  torch::Tensor src_input_embedding;

  torch::Tensor dst_tokens;
  torch::Tensor dst_positions;
  torch::Tensor dst_new_cache_slots;
  torch::Tensor dst_q_seq_lens;
  torch::Tensor dst_kv_seq_lens;
  torch::Tensor dst_q_cu_seq_lens;
  torch::Tensor dst_linear_state_indices;
  torch::Tensor dst_block_tables;
  torch::Tensor dst_input_embedding;

  int64_t padded_num_tokens = 0;
};

void model_input_buffer_updater(ModelInputBufferUpdaterParams& params);

bool has_model_input_buffer_updater_specialization(
    c10::ScalarType embedding_dtype,
    bool with_mrope,
    bool with_input_embedding,
    bool with_linear_state_indices,
    bool with_q_cu_seq_lens);

// Build merged mRoPE gather offsets for split_qkv_rmsnorm_mrope.
torch::Tensor build_split_qkv_rmsnorm_mrope_gather_pattern(
    int64_t rope_dim,
    const std::vector<int64_t>& mrope_section,
    bool is_interleaved,
    const torch::Device& device);

// Split fused [Q|K|V|G], apply q/k RMSNorm + mRoPE, and return
// (q, k, v, gate) on NPU.
//
// qkvg: [T, q_size + kv_size + kv_size + q_size] in Q | K | V | G layout.
// cos_sin: [T, 3 * rope_dim] with row layout
// [t_cos|t_sin|h_cos|h_sin|w_cos|w_sin].
// gather_pattern: merged gather offsets built by
// build_split_qkv_rmsnorm_mrope_gather_pattern(...).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
split_qkv_rmsnorm_mrope(const torch::Tensor& qkvg,
                        const torch::Tensor& q_weight,
                        const torch::Tensor& k_weight,
                        const torch::Tensor& cos_sin,
                        const torch::Tensor& gather_pattern,
                        float eps,
                        int64_t num_q_heads,
                        int64_t num_kv_heads,
                        int64_t head_size);

bool has_split_qkv_rmsnorm_mrope_specialization(int64_t num_q_heads,
                                                int64_t num_kv_heads,
                                                int64_t head_size);

}  // namespace xllm::kernel::npu::tilelang
