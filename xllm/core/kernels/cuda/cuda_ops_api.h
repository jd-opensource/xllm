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

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <glog/logging.h>

#include <optional>

#include "utils.h"

namespace xllm::kernel::cuda {

// TODO: add head_size parameter
void rotary_embedding(torch::Tensor& positions,
                      torch::Tensor& query,
                      std::optional<torch::Tensor> key,
                      torch::Tensor& cos_sin_cache,
                      // int64_t head_size,
                      bool is_neox);

// act_mode only support silu, gelu, gelu_tanh
void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode);

void reshape_paged_cache(
    torch::Tensor slot_ids,   // [n_tokens]
    torch::Tensor keys,       // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor values,     // [n_tokens, n_kv_heads, head_dim]
    torch::Tensor key_cache,  // [n_blocks, block_size, n_heads, head_dim]
    torch::Tensor value_cache);

void batch_prefill(const std::string& uri,
                   ffi::Array<int64_t> plan_info,
                   torch::Tensor float_workspace_buffer,
                   torch::Tensor int_workspace_buffer,
                   torch::Tensor page_locked_int_workspace_buffer,
                   torch::Tensor query,
                   torch::Tensor key,
                   torch::Tensor value,
                   torch::Tensor q_cu_seq_lens,
                   torch::Tensor kv_cu_seq_lens,
                   int64_t window_left,
                   double sm_scale,
                   torch::Tensor output,
                   std::optional<torch::Tensor>& output_lse,
                   bool enable_cuda_graph);

void batch_decode(const std::string& uri,
                  ffi::Array<int64_t> plan_info,
                  torch::Tensor float_workspace_buffer,
                  torch::Tensor int_workspace_buffer,
                  torch::Tensor page_locked_int_workspace_buffer,
                  torch::Tensor query,
                  torch::Tensor k_cache,
                  torch::Tensor v_cache,
                  torch::Tensor paged_kv_indptr,
                  torch::Tensor paged_kv_indices,
                  torch::Tensor paged_kv_last_page_len,
                  int64_t window_left,
                  double sm_scale,
                  torch::Tensor output,
                  std::optional<torch::Tensor>& output_lse,
                  bool enable_cuda_graph,
                  bool use_tensor_core,
                  torch::Tensor kv_seq_lens);

void rms_norm(torch::Tensor output,
              torch::Tensor input,
              torch::Tensor weight,
              double eps);

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon);

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias);

}  // namespace xllm::kernel::cuda
