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
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include "cuda_ops_api.h"
#include "utils.h"

namespace {

// Fused decoder reshape and cache kernel.
// Copies proj_k and proj_v into unshared_k_cache / unshared_v_cache at the
// positions specified by block_table and step.
// Inputs:
//   proj_k           : [batch_size, beam_size, kv_heads, head_dim]
//   proj_v           : [batch_size, beam_size, kv_heads, head_dim]
//   unshared_k_cache : [max_num_request, beam_size, max_decode_step, kv_heads,
//   head_dim] unshared_v_cache : [max_num_request, beam_size, max_decode_step,
//   kv_heads, head_dim] block_table      : [batch_size] - block_id per batch
//   step             : current decode step
template <typename scalar_t>
__global__ void decoder_reshape_and_cache_kernel(
    const scalar_t* __restrict__ proj_k,  // [batch_size, beam_size, kv_heads,
                                          // head_dim]
    const scalar_t* __restrict__ proj_v,  // [batch_size, beam_size, kv_heads,
                                          // head_dim]
    scalar_t* __restrict__ unshared_k_cache,  // [max_num_request, beam_size,
                                              // max_decode_step, kv_heads,
                                              // head_dim]
    scalar_t* __restrict__ unshared_v_cache,  // [max_num_request, beam_size,
                                              // max_decode_step, kv_heads,
                                              // head_dim]
    const int64_t* __restrict__ block_table,  // [batch_size]
    const int64_t batch_size,
    const int64_t beam_size,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int64_t max_decode_step,
    const int64_t max_num_request,
    const uint32_t step) {
  const int64_t total_elements = batch_size * beam_size * kv_heads;
  const int64_t idx = static_cast<int64_t>(blockIdx.y);

  if (idx >= total_elements) {
    return;
  }

  // Decode flattened index -> (batch, beam, kv_head)
  const int64_t batch_idx = idx / (beam_size * kv_heads);
  const int64_t remaining = idx % (beam_size * kv_heads);
  const int64_t beam_idx = remaining / kv_heads;
  const int64_t kv_head_idx = remaining % kv_heads;

  const int64_t block_id = block_table[batch_idx];

  // Guard invalid block id
  if (block_id < 0 || block_id >= max_num_request) {
    return;
  }

  // Compute base indices.
  // proj_k[batch_idx, beam_idx, kv_head_idx, :]
  const int64_t src_base =
      ((batch_idx * beam_size + beam_idx) * kv_heads + kv_head_idx) * head_dim;

  // unshared_*_cache[block_id, beam_idx, step, kv_head_idx, :]
  const int64_t dst_base =
      (((block_id * beam_size + beam_idx) * max_decode_step + step) * kv_heads +
       kv_head_idx) *
      head_dim;

  // Copy the full head_dim with threads parallelizing along D.
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    unshared_k_cache[dst_base + d] = proj_k[src_base + d];
    unshared_v_cache[dst_base + d] = proj_v[src_base + d];
  }
}

}  // namespace

namespace xllm::kernel::cuda {

void decoder_reshape_and_cache(torch::Tensor proj_k,
                               torch::Tensor proj_v,
                               torch::Tensor unshared_k_cache,
                               torch::Tensor unshared_v_cache,
                               torch::Tensor block_table,
                               uint32_t step) {
  TORCH_CHECK(proj_k.dim() == 4, "proj_k must be 4-dimensional");
  TORCH_CHECK(proj_v.dim() == 4, "proj_v must be 4-dimensional");
  TORCH_CHECK(unshared_k_cache.dim() == 5,
              "unshared_k_cache must be 5-dimensional");
  TORCH_CHECK(unshared_v_cache.dim() == 5,
              "unshared_v_cache must be 5-dimensional");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-dimensional");
  TORCH_CHECK(block_table.size(1) == 1, "block_table second dim must be 1");

  const int64_t batch_size = proj_k.size(0);
  const int64_t beam_size = proj_k.size(1);
  const int64_t kv_heads = proj_k.size(2);
  const int64_t head_dim = proj_k.size(3);
  const int64_t max_num_request = unshared_k_cache.size(0);
  const int64_t max_decode_step = unshared_k_cache.size(2);

  TORCH_CHECK(proj_v.sizes() == proj_k.sizes(),
              "proj_v and proj_k must have same shape");
  TORCH_CHECK(block_table.size(0) == batch_size,
              "block_table size must match batch_size");
  TORCH_CHECK(step >= 0 && step < max_decode_step,
              "step must be in valid range");
  TORCH_CHECK(unshared_k_cache.size(1) == beam_size,
              "unshared_k_cache beam_size mismatch");
  TORCH_CHECK(unshared_k_cache.size(3) == kv_heads,
              "unshared_k_cache kv_heads mismatch");
  TORCH_CHECK(unshared_k_cache.size(4) == head_dim,
              "unshared_k_cache head_dim mismatch");
  TORCH_CHECK(unshared_v_cache.sizes() == unshared_k_cache.sizes(),
              "unshared_v_cache and unshared_k_cache must have same shape");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(proj_k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor block_table_flat = block_table.select(1, 0).to(torch::kInt64);

  // Launch kernel: one block per (batch, beam, kv_head), threads along
  // head_dim.
  const int64_t total_elements = batch_size * beam_size * kv_heads;
  const int threads_per_block = 128;
  dim3 block_dim(threads_per_block, 1, 1);
  dim3 grid_dim(1, static_cast<unsigned int>(total_elements), 1);

  DISPATCH_FLOATING_TYPES(
      proj_k.scalar_type(), "decoder_reshape_and_cache_kernel", [&] {
        decoder_reshape_and_cache_kernel<scalar_t>
            <<<grid_dim, block_dim, 0, stream>>>(
                proj_k.data_ptr<scalar_t>(),
                proj_v.data_ptr<scalar_t>(),
                unshared_k_cache.data_ptr<scalar_t>(),
                unshared_v_cache.data_ptr<scalar_t>(),
                block_table_flat.data_ptr<int64_t>(),
                batch_size,
                beam_size,
                kv_heads,
                head_dim,
                max_decode_step,
                max_num_request,
                step);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda