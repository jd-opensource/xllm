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

#include "ATen/Tensor.h"

namespace xllm::mlu {

void flash_attention(const at::Tensor& q,
                     const at::Tensor& k,
                     const at::Tensor& v,
                     const at::Tensor& out,
                     const c10::optional<at::Tensor>& output_lse,
                     const c10::optional<at::Tensor>& cu_seq_lens_q,
                     const c10::optional<at::Tensor>& cu_seq_lens_kv,
                     const c10::optional<at::Tensor>& alibi_slope,
                     const c10::optional<at::Tensor>& attn_bias,
                     const c10::optional<at::Tensor>& q_quant_scale,
                     const c10::optional<at::Tensor>& k_cache_quant_scale,
                     const c10::optional<at::Tensor>& v_cache_quant_scale,
                     const c10::optional<at::Tensor>& out_quant_scale,
                     const c10::optional<at::Tensor>& block_tables,
                     const int64_t max_seq_len_q,
                     const int64_t max_seq_len_kv,
                     const double softmax_scale,
                     const bool is_causal,
                     const int64_t window_size_left,
                     const int64_t window_size_right,
                     const std::string& compute_dtype,
                     bool return_lse);

void single_query_cached_kv_attn(
    const torch::Tensor& q_ori,
    const torch::Tensor& k_cache,
    const torch::Tensor& output,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,  // [batch]
    const c10::optional<torch::Tensor>& v_cache,
    const c10::optional<torch::Tensor>& output_lse,
    const c10::optional<torch::Tensor>& q_quant_scale,
    const c10::optional<torch::Tensor>& k_cache_quant_scale,
    const c10::optional<torch::Tensor>& v_cache_quant_scale,
    const c10::optional<torch::Tensor>& out_quant_scale,
    const c10::optional<torch::Tensor>&
        alibi_slopes,  // [bs, head_num] or [head_num]
    const std::string& compute_dtype,
    int64_t max_context_len,
    int64_t windows_size_left,
    int64_t windows_size_right,
    double softmax_scale,
    bool return_lse,
    int64_t kv_cache_quant_bit_size);

void reshape_paged_cache(torch::Tensor& k,
                         const c10::optional<torch::Tensor>& v,
                         torch::Tensor& k_cache,
                         const c10::optional<torch::Tensor>& v_cache,
                         const torch::Tensor& slot_mapping,
                         bool direction);

}  // namespace xllm::mlu
