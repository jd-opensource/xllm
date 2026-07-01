/* Copyright 2025-2026 The xLLM Authors.

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

#if defined(USE_DCU)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <cstdint>

namespace xllm::kernel::cuda {

#if defined(USE_DCU)
using LlmDecodeMetadataUpdateStream = hipStream_t;
#else
using LlmDecodeMetadataUpdateStream = cudaStream_t;
#endif

struct LlmDecodeMetadataUpdateParams {
  const int32_t* src_tokens;
  const int32_t* src_positions;
  const int32_t* src_new_cache_slots;
  const int32_t* src_kv_seq_lens;
  const int32_t* src_paged_kv_indptr;
  const int32_t* src_paged_kv_indices;
  const int32_t* src_paged_kv_last_page_len;
  int32_t* dst_tokens;
  int32_t* dst_positions;
  int32_t* dst_new_cache_slots;
  int32_t* dst_kv_seq_lens;
  int32_t* dst_kv_seq_lens_delta;
  int32_t* dst_paged_kv_indptr;
  int32_t* dst_paged_kv_indices;
  int32_t* dst_paged_kv_last_page_len;
  int64_t actual_num_tokens;
  int64_t padded_num_tokens;
  int64_t actual_batch_size;
  // Static upper-bound on the paged_kv_indices count for the current decode
  // batch. Used only on the host side to size launch geometry; the kernel
  // reads the actual indices count dynamically from
  // src_paged_kv_indptr[actual_batch_size] so CUDA-graph replay sees the
  // correct count even when the KV cache crosses a block boundary between
  // capture and replay.
  int64_t actual_indices_size;
  // Worst-case paged_kv_indices count this captured graph instance will ever
  // need to handle (typically `max_seqs_per_batch * max_block_table_len`).
  // 0 means "treat as eager mode" -- launcher falls back to
  // `actual_indices_size` for grid sizing. When non-zero, the launcher uses
  // this as the kernel's `max_work_size` so the captured strided loop
  // iterates far enough to cover any runtime indices growth.
  int64_t max_indices_size_for_graph_capacity;
};

void update_llm_decode_metadata(const LlmDecodeMetadataUpdateParams& params,
                                LlmDecodeMetadataUpdateStream stream);

}  // namespace xllm::kernel::cuda
