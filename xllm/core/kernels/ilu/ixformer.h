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
at::Tensor ixinfer_flash_attn_unpad(
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& value,
    at::Tensor& out,
    at::Tensor& cu_seq_q,
    at::Tensor& cu_seq_k,
    int64_t max_seq_q,
    int64_t max_seq_k,
    bool is_causal,
    int64_t window_left,
    int64_t window_right,
    double scale,
    double softcap,
    bool sqrt_alibi,
    const c10::optional<at::Tensor>& alibi_slopes,
    const c10::optional<at::Tensor>& sinks,
    c10::optional<at::Tensor>& lse);

void silu_and_mul(at::Tensor& input, at::Tensor& output);

at::Tensor vllm_paged_attention(at::Tensor& out,
                                at::Tensor& query,
                                at::Tensor& key_cache,
                                at::Tensor& value_cache,
                                int64_t num_kv_heads,
                                double scale,
                                at::Tensor& block_tables,
                                at::Tensor& context_lens,
                                int64_t block_size,
                                int64_t max_context_len,
                                const c10::optional<at::Tensor>& alibi_slopes,
                                bool causal,
                                int window_left,
                                int window_right,
                                double softcap,
                                bool enable_cuda_graph,
                                bool use_sqrt_alibi,
                                const c10::optional<at::Tensor>& sinks);

void vllm_reshape_and_cache(at::Tensor& key,
                            at::Tensor& value,
                            at::Tensor& key_cache,
                            at::Tensor& value_cache,
                            at::Tensor& slot_mapping,
                            int64_t key_token_stride,
                            int64_t value_token_stride);

void vllm_rotary_embedding(at::Tensor& positions,
                           at::Tensor& query,
                           at::Tensor& key,
                           int64_t head_size,
                           at::Tensor& cos_sin_cache,
                           bool is_neox);

void residual_layer_norm(at::Tensor& input,
                         at::Tensor& residual,
                         at::Tensor& weight,
                         at::Tensor& bias,
                         c10::optional<at::Tensor>& fused_bias,
                         c10::optional<at::Tensor>& output,
                         c10::optional<at::Tensor>& residual_output,
                         double alpha,
                         double eps,
                         bool is_post);
}  // namespace ixformer::infer
