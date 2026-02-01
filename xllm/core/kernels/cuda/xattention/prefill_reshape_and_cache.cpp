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

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "cuda.h"

namespace xllm::kernel::cuda {

// Prefill reshape and cache kernel for Rec multi-round mode.
// Copies projected K and V tensors from prefill phase into shared KV cache.
// This function is specifically designed for Rec (recommendation) multi-round
// mode, where multi-round decode loops run on device. In this mode, sequences
// share the same prompt prefix, and the prefill KV cache needs to be stored in
// a shared cache format for efficient multi-round decoding on device. Inputs:
//   proj_k          : [shared_len, kv_heads, head_dim] - projected K tensor
//                     from prefill phase
//   proj_v          : [shared_len, kv_heads, head_dim] - projected V tensor
//                     from prefill phase
//   shared_k_cache  : [num_shared_kv_seq_len, kv_heads, head_dim] - shared K
//                     cache buffer for storing prefill KV cache
//   shared_v_cache  : [num_shared_kv_seq_len, kv_heads, head_dim] - shared V
//                     cache buffer for storing prefill KV cache
void prefill_reshape_and_cache(
    torch::Tensor proj_k,  // [shared_len, kv_heads, head_dim]
    torch::Tensor proj_v,  // [shared_len, kv_heads, head_dim]
    torch::Tensor
        shared_k_cache,  // [num_shared_kv_seq_len, kv_heads, head_dim]
    torch::Tensor shared_v_cache) {
  int64_t shared_len = proj_k.size(0);
  shared_k_cache = shared_k_cache.slice(0, 0, shared_len);
  shared_v_cache = shared_v_cache.slice(0, 0, shared_len);
  shared_k_cache.copy_(proj_k);
  shared_v_cache.copy_(proj_v);
}

}  // namespace xllm::kernel::cuda
