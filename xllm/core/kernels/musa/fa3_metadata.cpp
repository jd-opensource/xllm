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

// Precomputes the scheduler_metadata tensor consumed by the FA3 decode
// kernel (fa3_fwd.cpp). The tensor has shape [batch_size * 4] int32 and is
// produced by the JIT-built `fmha_get_metadata_<dispatch>.so` from mate's
// cached_ops, mirroring _fmha_get_metadata() in mate/jit/attention/fmha/.
//
// We call this once per shape (per PlanInfo) at layer 0, then reuse the
// returned tensor for every FA3 decode call across all attention layers.

#include <glog/logging.h>

#include <cstdint>
#include <string>

#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {

namespace {

// Hard-coded for Qwen3.5-27B full-attn decode prototype:
//   head_ratio=6, packgqa=true, ragged_q (cu_seqlens_q provided),
//   padded_k (no cu_seqlens_k), causal, num_warps=1 (batch_size <= 31).
// Config-driven dispatch is a follow-up.
constexpr const char* kFa3MetadataUri =
    "fmha_get_metadata_6x1_ragged_q_padded_k_causal_packgqa";

// Tile sizes the FA3 kernel was JIT-built with for our hash. Must match the
// `TileM` / `TileN` constexpr inside the fmha_fwd_<hash>.so so the metadata
// kernel produces a schedule that the run kernel can execute.
constexpr int32_t kFa3TileM = 32;
constexpr int32_t kFa3TileN = 64;

}  // namespace

// Returns a fresh [batch_size * 4] int32 device tensor populated with the
// FA3 scheduler metadata. Caller stores the tensor in PlanInfo and reuses
// it across all decode layers for the same shape.
torch::Tensor fa3_decode_scheduler_metadata(
    const torch::Device& device,
    int32_t batch_size,
    int32_t num_heads_q,
    int32_t num_heads_kv,
    int32_t head_dim_qk,
    int32_t head_dim_vo,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t window_size_left,
    int32_t window_size_right,
    const torch::Tensor& cu_seqlens_q,  // [batch+1] int32 device
    const torch::Tensor& seqused_k      // [batch]   int32 device
) {
  CHECK_GT(batch_size, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);

  // Flat 4-slot-per-batch int32 metadata buffer, layout matches mate's
  // _fmha_get_metadata: [num_splits_dynamic | batch_table | num_m_blocks |
  //                      num_nheads_in_l2], each of length batch_size.
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor metadata =
      torch::empty({static_cast<int64_t>(batch_size) * 4}, options);

  MusaTvmffiStreamGuard stream_guard(device);

  // The kernel writes into 4 separate slices of `metadata`. Slice into
  // contiguous views (they share storage; the kernel sees device pointers).
  const int64_t b = batch_size;
  auto num_splits_dynamic = metadata.slice(/*dim=*/0, /*start=*/0, /*end=*/b);
  auto batch_table         = metadata.slice(/*dim=*/0, /*start=*/b, /*end=*/2 * b);
  auto num_m_blocks        = metadata.slice(/*dim=*/0, /*start=*/2 * b, /*end=*/3 * b);
  auto num_nheads_in_l2    = metadata.slice(/*dim=*/0, /*start=*/3 * b, /*end=*/4 * b);

  const std::string uri = kFa3MetadataUri;

  // Match the kernel signature in
  // /root/.cache/mate/0.2.2/mp31/generated/fmha_get_metadata_*.mu:
  //   int batch_size, num_heads_q, num_heads_kv, headdim, headdim_v,
  //   int max_seqlen_q, max_seqlen_k, max_seqlen_k_new,
  //   Optional<Tensor> cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
  //                    cu_seqlens_k_new,
  //   int window_size_left, window_size_right,
  //   Optional<Tensor> leftpad_k,
  //   Tensor num_splits_dynamic, batch_table, num_m_blocks, num_nheads_in_l2,
  //   int num_splits, TileM, TileN, mp_margin.
  get_function(uri, uri)(
      static_cast<int64_t>(batch_size),
      static_cast<int64_t>(num_heads_q),
      static_cast<int64_t>(num_heads_kv),
      static_cast<int64_t>(head_dim_qk),
      static_cast<int64_t>(head_dim_vo),
      static_cast<int64_t>(max_seqlen_q),
      static_cast<int64_t>(max_seqlen_k),
      /*max_seqlen_k_new=*/static_cast<int64_t>(0),
      to_ffi_tensor(cu_seqlens_q),                  // cu_seqlens_q
      ffi::Optional<ffi::Tensor>(),                 // cu_seqlens_k
      ffi::Optional<ffi::Tensor>(),                 // seqused_q
      to_ffi_tensor(seqused_k),                     // seqused_k
      ffi::Optional<ffi::Tensor>(),                 // cu_seqlens_k_new
      static_cast<int64_t>(window_size_left),
      static_cast<int64_t>(window_size_right),
      ffi::Optional<ffi::Tensor>(),                 // leftpad_k
      to_ffi_tensor(num_splits_dynamic),
      to_ffi_tensor(batch_table),
      to_ffi_tensor(num_m_blocks),
      to_ffi_tensor(num_nheads_in_l2),
      /*num_splits=*/static_cast<int64_t>(1),
      static_cast<int64_t>(kFa3TileM),
      static_cast<int64_t>(kFa3TileN),
      /*mp_margin=*/static_cast<int64_t>(0));

  return metadata;
}

}  // namespace xllm::kernel::cuda
