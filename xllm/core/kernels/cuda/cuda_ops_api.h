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

#include "tvm_ffi_utils.h"

namespace xllm::kernel::cuda {

void apply_rope_pos_ids_cos_sin_cache(TensorView q,
                                      TensorView k,
                                      TensorView q_rope,
                                      TensorView k_rope,
                                      TensorView cos_sin_cache,
                                      TensorView pos_ids,
                                      bool interleave);

void act_and_mul(TensorView out,
                 TensorView input,
                 const std::string& act_mode,
                 bool enable_pdl);

void reshape_paged_cache(
    const torch::Tensor& slot_ids,  // [n_tokens]
    const torch::Tensor& keys,      // [n_tokens, n_kv_heads, head_dim]
    const torch::Tensor& values,    // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor& key_cache,       // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor& value_cache);

void batch_prefill(torch::Tensor& float_workspace_buffer,
                   torch::Tensor& int_workspace_buffer,
                   torch::Tensor& page_locked_int_workspace_buffer,
                   const torch::Tensor& query,
                   const torch::Tensor& key,
                   const torch::Tensor& value,
                   const torch::Tensor& q_cu_seq_lens,
                   const torch::Tensor& kv_cu_seq_lens,
                   int64_t window_size_left,
                   torch::Tensor& output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph,
                   bool enable_pdl);

void batch_decode(torch::Tensor& float_workspace_buffer,
                  torch::Tensor& int_workspace_buffer,
                  torch::Tensor& page_locked_int_workspace_buffer,
                  const torch::Tensor& query,
                  const torch::Tensor& k_cache,
                  const torch::Tensor& v_cache,
                  const torch::Tensor& q_cu_seq_lens,
                  const torch::Tensor& paged_kv_indptr,
                  const torch::Tensor& paged_kv_indices,
                  const torch::Tensor& paged_kv_last_page_len,
                  int64_t window_size_left,
                  torch::Tensor& output,
                  std::optional<torch::Tensor>& output_lse,
                  bool enable_cuda_graph,
                  bool enable_pdl);

void rmsnorm(TensorView output,
             TensorView input,
             TensorView weight,
             double eps,
             bool enable_pdl);

torch::Tensor matmul(const torch::Tensor& a,
                     const torch::Tensor& b,
                     const std::optional<torch::Tensor>& bias);

}  // namespace xllm::kernel::cuda