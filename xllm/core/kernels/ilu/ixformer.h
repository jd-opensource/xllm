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
#include <torch/all.h>

#include "ATen/Tensor.h"
#include "utils.h"

namespace ixformer::infer {
torch::Tensor ixinfer_flash_attn_unpad_with_block_tables(
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& out,
    torch::Tensor& block_tables,
    torch::Tensor& cu_seq_q,
    torch::Tensor& cu_seq_k,
    int64_t max_seq_q,
    int64_t max_seq_k,
    bool is_causal,
    int64_t window_left,
    int64_t window_right,
    double scale,
    double softcap,
    bool sqrt_alibi,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::optional<torch::Tensor>& sinks,
    std::optional<torch::Tensor>& lse);

void silu_and_mul(torch::Tensor& input, torch::Tensor& output);

torch::Tensor vllm_paged_attention(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    bool causal,
    int32_t window_left,
    int32_t window_right,
    double softcap,
    bool enable_cuda_graph,
    bool use_sqrt_alibi,
    const std::optional<torch::Tensor>& sinks);

void vllm_reshape_and_cache(torch::Tensor& key,
                            torch::Tensor& value,
                            torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping,
                            int64_t key_token_stride,
                            int64_t value_token_stride);

void vllm_rotary_embedding(torch::Tensor& positions,
                           torch::Tensor& query,
                           torch::Tensor& key,
                           int64_t head_size,
                           torch::Tensor& cos_sin_cache,
                           bool is_neox);

void residual_rms_norm(torch::Tensor& input,
                       torch::Tensor& residual,
                       torch::Tensor& weight,
                       std::optional<torch::Tensor>& output,
                       std::optional<torch::Tensor>& residual_output,
                       std::optional<torch::Tensor>& fused_bias,
                       double alpha,
                       double eps,
                       bool is_post);

void rms_norm(torch::Tensor& input,
              torch::Tensor& weight,
              torch::Tensor& output,
              std::optional<torch::Tensor>& fused_bias,
              double eps);

}  // namespace ixformer::infer
