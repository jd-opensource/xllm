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
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/cuda.h>

#include "kernels/cuda/utils.h"
#include "xattention_ops_api.h"

namespace {

// decoder reshape and cache kernel.
// Copies proj_k and proj_v into unshared_k_cache / unshared_v_cache.
// Inputs:
//   proj_k           : [batch_size, beam_size, kv_heads, head_dim]
//   proj_v           : [batch_size, beam_size, kv_heads, head_dim]
//   step             : [1] - current decode step
//   batch_size       : batch size
//   beam_size        : beam size
//   kv_heads         : number of kv heads
//   head_dim         : head dimension
//   k_stride0        : proj_k.stride(0)
//   k_stride1        : proj_k.stride(1)
//   v_stride0        : proj_v.stride(0)
//   v_stride1        : proj_v.stride(1)
//   cache_stride0    : unshared_k_cache.stride(0)
//   cache_stride1    : unshared_k_cache.stride(1)
//   cache_stride2    : unshared_k_cache.stride(2)
//   cache_stride3    : unshared_k_cache.stride(3)
// Outputs:
//   unshared_k_cache : [max_batch_size, beam_size, max_step, kv_heads,
//   head_dim]
//   unshared_v_cache : [max_batch_size, beam_size, max_step, kv_heads,
//   head_dim]

template <typename scalar_t>
__global__ void decoder_reshape_and_cache_kernel(
    const scalar_t* __restrict__ proj_k,
    const scalar_t* __restrict__ proj_v,
    scalar_t* __restrict__ unshared_k_cache,
    scalar_t* __restrict__ unshared_v_cache,
    const int32_t* __restrict__ step,
    const int64_t batch_size,
    const int64_t beam_size,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int64_t k_stride0,
    const int64_t k_stride1,
    const int64_t v_stride0,
    const int64_t v_stride1,
    const int64_t cache_stride0,
    const int64_t cache_stride1,
    const int64_t cache_stride2,
    const int64_t cache_stride3) {
  const int64_t total_elements = batch_size * beam_size * kv_heads;
  const int64_t idx = static_cast<int64_t>(blockIdx.y);

  if (idx >= total_elements) {
    return;
  }

  // Decode flattened index -> (batch_idx, beam_idx, kv_head_idx)
  const int64_t batch_idx = idx / (beam_size * kv_heads);
  const int64_t remaining = idx % (beam_size * kv_heads);
  const int64_t beam_idx = remaining / kv_heads;
  const int64_t kv_head_idx = remaining % kv_heads;

  // Read step value from tensor
  const int64_t current_step = *step;

  // Compute source indices using stride
  // proj_k[batch_idx, beam_idx, kv_head_idx, :]
  // Note: proj_k is [batch_size, beam_size, kv_heads, head_dim]
  // We need stride(2) for kv_heads dimension, but assume it's head_dim for
  // simplicity If non-contiguous, should add k_stride2 parameter
  const int64_t k_src_base =
      batch_idx * k_stride0 + beam_idx * k_stride1 + kv_head_idx * head_dim;
  const int64_t v_src_base =
      batch_idx * v_stride0 + beam_idx * v_stride1 + kv_head_idx * head_dim;

  // Compute destination indices using stride
  // unshared_k_cache[batch_idx, beam_idx, current_step, kv_head_idx, :]
  const int64_t cache_dst_base =
      batch_idx * cache_stride0 + beam_idx * cache_stride1 +
      current_step * cache_stride2 + kv_head_idx * cache_stride3;

  // Copy the full head_dim with threads parallelizing along D.
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    unshared_k_cache[cache_dst_base + d] = proj_k[k_src_base + d];
    unshared_v_cache[cache_dst_base + d] = proj_v[v_src_base + d];
  }
}

}  // namespace

namespace xllm::kernel::cuda {

void decoder_reshape_and_cache(torch::Tensor proj_k,
                               torch::Tensor proj_v,
                               torch::Tensor unshared_k_cache,
                               torch::Tensor unshared_v_cache,
                               torch::Tensor step) {
  CHECK_EQ(proj_k.dim(), 4) << "proj_k must be 4-dimensional";
  CHECK_EQ(proj_v.dim(), 4) << "proj_v must be 4-dimensional";
  CHECK_EQ(unshared_k_cache.dim(), 5)
      << "unshared_k_cache must be 5-dimensional";
  CHECK_EQ(unshared_v_cache.dim(), 5)
      << "unshared_v_cache must be 5-dimensional";

  const int64_t batch_size = proj_k.size(0);
  const int64_t beam_size = proj_k.size(1);
  const int64_t kv_heads = proj_k.size(2);
  const int64_t head_dim = proj_k.size(3);

  CHECK_EQ(proj_v.sizes(), proj_k.sizes())
      << "proj_v and proj_k must have same shape";
  CHECK_EQ(unshared_k_cache.size(3), kv_heads)
      << "unshared_k_cache kv_heads mismatch";
  CHECK_EQ(unshared_k_cache.size(4), head_dim)
      << "unshared_k_cache head_dim mismatch";
  CHECK(unshared_v_cache.sizes() == unshared_k_cache.sizes())
      << "unshared_v_cache and unshared_k_cache must have same shape";

  const at::cuda::OptionalCUDAGuard device_guard(device_of(proj_k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Get stride information for non-contiguous tensor support
  const int64_t k_stride0 = proj_k.stride(0);
  const int64_t k_stride1 = proj_k.stride(1);
  const int64_t v_stride0 = proj_v.stride(0);
  const int64_t v_stride1 = proj_v.stride(1);
  const int64_t cache_stride0 = unshared_k_cache.stride(0);
  const int64_t cache_stride1 = unshared_k_cache.stride(1);
  const int64_t cache_stride2 = unshared_k_cache.stride(2);
  const int64_t cache_stride3 = unshared_k_cache.stride(3);

  // Launch kernel: one block per (token, kv_head), threads along head_dim.
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
                step.data_ptr<int32_t>(),
                batch_size,
                beam_size,
                kv_heads,
                head_dim,
                k_stride0,
                k_stride1,
                v_stride0,
                v_stride1,
                cache_stride0,
                cache_stride1,
                cache_stride2,
                cache_stride3);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
