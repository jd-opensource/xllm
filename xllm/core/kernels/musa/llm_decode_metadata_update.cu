/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>

#include "core/kernels/musa/llm_decode_metadata_update.h"

namespace xllm::kernel::cuda {
namespace {

constexpr int32_t kThreadsPerBlock = 256;
constexpr int64_t kMaxBlocksPerLaunch = 4096;

__global__ void llm_decode_metadata_update_kernel(
    LlmDecodeMetadataUpdateParams params,
    int64_t max_work_size) {
  const int64_t thread_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;

  // Read the runtime paged-KV indices count from device memory rather than
  // using `params.actual_indices_size` (which is baked into the kernel's
  // parameter buffer at CUDA-graph capture time). At capture, the warmup
  // batch typically uses 1 KV-cache block per sequence, so the baked
  // `actual_indices_size` is 1. When the captured graph is later replayed at
  // a decode step that has crossed a block boundary (e.g., position 64 with
  // block_size=64), the actual indices count grows to 2+, but the captured
  // kernel would still copy only 1 entry, leaving subsequent entries in
  // `dst_paged_kv_indices` stale from warmup. The attention kernel that
  // reads `dst_paged_kv_indices[0..indptr[bs]-1]` then dereferences a wrong
  // block ID and produces silently-wrong attention output starting at the
  // first full-attention layer. Read the count from indptr so the loop
  // bound tracks the runtime KV cache layout instead.
  const int64_t dyn_indices_size =
      (params.src_paged_kv_indptr != nullptr && params.actual_batch_size > 0)
          ? static_cast<int64_t>(
                params.src_paged_kv_indptr[params.actual_batch_size])
          : params.actual_indices_size;

  for (int64_t idx = thread_idx; idx < max_work_size; idx += step) {
    if (idx < params.actual_num_tokens) {
      params.dst_tokens[idx] = params.src_tokens[idx];
      params.dst_positions[idx] = params.src_positions[idx];
      params.dst_new_cache_slots[idx] = params.src_new_cache_slots[idx];
    }
    if (idx >= params.actual_num_tokens && idx < params.padded_num_tokens) {
      params.dst_tokens[idx] = 0;
      params.dst_new_cache_slots[idx] = 0;
    }
    if (idx < params.actual_batch_size + 1) {
      params.dst_kv_seq_lens[idx] = params.src_kv_seq_lens[idx];
      params.dst_paged_kv_indptr[idx] = params.src_paged_kv_indptr[idx];
    }
    if (idx < params.actual_batch_size) {
      params.dst_kv_seq_lens_delta[idx] =
          params.src_kv_seq_lens[idx + 1] - params.src_kv_seq_lens[idx];
      params.dst_paged_kv_last_page_len[idx] =
          params.src_paged_kv_last_page_len[idx];
    }
    if (idx < dyn_indices_size) {
      params.dst_paged_kv_indices[idx] = params.src_paged_kv_indices[idx];
    }
  }
}

}  // namespace

void update_llm_decode_metadata(const LlmDecodeMetadataUpdateParams& params,
                                LlmDecodeMetadataUpdateStream stream) {
  int64_t max_work_size = params.actual_num_tokens;
  if (params.padded_num_tokens > max_work_size) {
    max_work_size = params.padded_num_tokens;
  }
  if (params.actual_batch_size + 1 > max_work_size) {
    max_work_size = params.actual_batch_size + 1;
  }
  if (params.actual_indices_size > max_work_size) {
    max_work_size = params.actual_indices_size;
  }
  // When this kernel is invoked under CUDA-graph capture, the
  // `max_work_size` value is baked into the captured launch's grid geometry
  // and loop bound. The runtime paged_kv_indices count can grow between
  // capture and replay (block boundary crossing during decode); use the
  // caller-provided worst-case capacity so the captured strided loop
  // iterates far enough at replay time to cover the grown indices array.
  // Callers that don't run under graph capture leave this 0 and the
  // ordinary actual_* sizing wins.
  if (params.max_indices_size_for_graph_capacity > max_work_size) {
    max_work_size = params.max_indices_size_for_graph_capacity;
  }
  if (max_work_size <= 0) {
    return;
  }
  // Cap the grid size because the kernel already uses a strided loop.
  // This keeps launch overhead bounded for large inputs without reducing
  // coverage.
  const int64_t num_blocks = std::min<int64_t>(
      (max_work_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
      kMaxBlocksPerLaunch);
  llm_decode_metadata_update_kernel<<<static_cast<uint32_t>(num_blocks),
                                      kThreadsPerBlock,
                                      /*shared_mem_bytes=*/0,
                                      stream>>>(params, max_work_size);
  const cudaError_t error = cudaGetLastError();
  CHECK_EQ(error, cudaSuccess)
      << "llm_decode_metadata_update kernel launch failed: "
      << cudaGetErrorString(error);
}

}  // namespace xllm::kernel::cuda
