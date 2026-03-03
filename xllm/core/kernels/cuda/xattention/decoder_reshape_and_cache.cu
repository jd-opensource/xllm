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
#include <glog/logging.h>
#include <torch/cuda.h>

#include "kernels/cuda/utils.h"
#include "xattention_ops_api.h"

namespace {

// Fused decoder reshape and cache kernel.
// Copies proj_k and proj_v into unshared_k_cache / unshared_v_cache at the
// positions specified by block_table and step.
// Inputs:
//   proj_k           : [num_seqs, kv_heads, head_dim] (3D, num_seqs =
//   batch_size * beam_size) proj_v           : [num_seqs, kv_heads, head_dim]
//   unshared_k_cache : [max_num_seqs, max_decode_step, kv_heads, head_dim] (4D)
//   unshared_v_cache : [max_num_seqs, max_decode_step, kv_heads, head_dim] (4D)
//   block_table      : [num_seqs, 1] - block_id per sequence
//   step             : [1] - current decode step tensor
template <typename scalar_t>
__global__ void decoder_reshape_and_cache_kernel(
    const scalar_t* __restrict__ proj_k,  // [num_seqs, kv_heads, head_dim]
    const scalar_t* __restrict__ proj_v,
    scalar_t* __restrict__ unshared_k_cache,  // [max_num_seqs, max_decode_step,
                                              // kv_heads, head_dim]
    scalar_t* __restrict__ unshared_v_cache,
    const int32_t* __restrict__ block_table,  // [num_seqs, 1]
    const int32_t* __restrict__ step,         // [1] - current decode step
    const int64_t block_table_stride,
    const int64_t num_seqs,
    const int64_t kv_heads,
    const int64_t head_dim,
    const int64_t max_decode_step,
    const int64_t max_num_seqs) {
  const int64_t total_elements = num_seqs * kv_heads;
  const int64_t idx = static_cast<int64_t>(blockIdx.y);

  if (idx >= total_elements) {
    return;
  }

  // Decode flattened index -> (seq_idx, kv_head)
  const int64_t seq_idx = idx / kv_heads;
  const int64_t kv_head_idx = idx % kv_heads;

  const int32_t block_id = block_table[seq_idx * block_table_stride];

  // Guard invalid block id
  if (block_id < 0 || block_id >= max_num_seqs) {
    return;
  }

  // Read step value from tensor
  const int64_t current_step = *step;

  // Compute base indices
  // proj_k[seq_idx, kv_head_idx, :]
  const int64_t src_base = (seq_idx * kv_heads + kv_head_idx) * head_dim;

  // unshared_*_cache[block_id, step, kv_head_idx, :]
  const int64_t dst_base =
      ((block_id * max_decode_step + current_step) * kv_heads + kv_head_idx) *
      head_dim;

  // Copy the full head_dim with threads parallelizing along D
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
                               torch::Tensor step) {
  CHECK_EQ(proj_k.dim(), 3) << "proj_k must be 3-dimensional";
  CHECK_EQ(proj_v.dim(), 3) << "proj_v must be 3-dimensional";
  CHECK_EQ(unshared_k_cache.dim(), 5)
      << "unshared_k_cache must be 5-dimensional";
  CHECK_EQ(unshared_v_cache.dim(), 5)
      << "unshared_v_cache must be 5-dimensional";
  CHECK_EQ(block_table.dim(), 2) << "block_table must be 2-dimensional";
  CHECK_EQ(block_table.size(1), 1) << "block_table second dim must be 1";
  CHECK_EQ(step.dim(), 1) << "step must be 1-dimensional";
  CHECK_EQ(step.size(0), 1) << "step must have shape [1]";

  const int64_t num_seqs = proj_k.size(0);  // batch_size * beam_size
  const int64_t kv_heads = proj_k.size(1);
  const int64_t head_dim = proj_k.size(2);
  const int64_t max_num_seqs =
      unshared_k_cache.size(0) * unshared_k_cache.size(1);
  const int64_t max_decode_step = unshared_k_cache.size(2);

  CHECK(proj_v.sizes() == proj_k.sizes())
      << "proj_v and proj_k must have same shape";
  CHECK_EQ(block_table.size(0), num_seqs)
      << "block_table size must match num_seqs";
  CHECK_EQ(unshared_k_cache.size(3), kv_heads)
      << "unshared_k_cache kv_heads mismatch";
  CHECK_EQ(unshared_k_cache.size(4), head_dim)
      << "unshared_k_cache head_dim mismatch";
  CHECK(unshared_v_cache.sizes() == unshared_k_cache.sizes())
      << "unshared_v_cache and unshared_k_cache must have same shape";

  const at::cuda::OptionalCUDAGuard device_guard(device_of(proj_k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::Tensor block_table_i32 = block_table.scalar_type() == torch::kInt32
                                      ? block_table
                                      : block_table.to(torch::kInt32);
  const int64_t block_table_stride = block_table_i32.stride(0);

  // Launch kernel: one block per (seq, kv_head), threads along head_dim
  const int64_t total_elements = num_seqs * kv_heads;
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
                block_table_i32.data_ptr<int32_t>(),
                step.data_ptr<int32_t>(),
                block_table_stride,
                num_seqs,
                kv_heads,
                head_dim,
                max_decode_step,
                max_num_seqs);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace xllm::kernel::cuda
