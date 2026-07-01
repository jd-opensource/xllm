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

// FA3 decode dispatch via the cached `fmha_fwd_<hash>.so` from mate's JIT
// cache. Targets the FA3 unified attention kernel (single-pass, warp-
// specialized) instead of the FlashInfer fa2 split-KV BatchDecode that
// `batch_decode.cpp` uses.
//
// Wire-up: `flashinfer_attention.cpp::decoder_forward` selects this path
// when env var XLLM_USE_FA3=1, behind the same enable_cuda_graph rules
// as the legacy path.
//
// Kernel ABI: 39 positional args, signature defined in
//   /root/.cache/mate/0.2.2/mp31/generated/fmha_fwd_<hash>.mu
// Scheduler metadata is split into three Optional<Tensor> slices
// (num_splits_dynamic | batch_table | num_m_blocks), each of length
// `batch_size`. The slices are views into the flat tensor produced by
// fa3_decode_scheduler_metadata() (which writes 4 stripes; we use the
// first 3 here, the 4th is consumed elsewhere/unused for this config).
//
// The hash is identical between MUSA 4.3.5 and 5.1.0 because it is
// computed from the constexpr config dict alone; the .so itself differs
// (MUSA SDK ABI).

#include <glog/logging.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {

namespace {

// Hard-coded for the prototype. Config-driven hash lookup is a follow-up.
// This corresponds to:
//   HeadDimQK=256, HeadDimVO=256, HeadRatio=6, PagedKV=true, IsCausal=true,
//   HasCuseqlensQ=true (ragged Q), HasSequsedK=true (per-seq KV length),
//   ForceLsuKV=false.
// Suitable for both prefill and decode with ragged Q layout (xllm always
// uses ragged Q via q_cu_seq_lens).
constexpr const char* kFa3FwdUriHash =
    "9e4f4b2e6574a7a45a93fef39cf9b0485651e39052d9dfd88c2e1439137a9374";

std::string fa3_fwd_uri() {
  return std::string("fmha_fwd_") + kFa3FwdUriHash;
}

ffi::Optional<ffi::Tensor> none_tensor() {
  return ffi::Optional<ffi::Tensor>();
}

ffi::Optional<int64_t> none_int() {
  return ffi::Optional<int64_t>();
}

}  // namespace

void fa3_decode(const torch::Tensor& query,
                const torch::Tensor& k_cache,
                const torch::Tensor& v_cache,
                const torch::Tensor& cu_seqlens_q,
                const torch::Tensor& seqused_k,
                const torch::Tensor& page_table,
                const torch::Tensor& scheduler_metadata,
                int64_t max_seqlen_q,
                int64_t window_left,
                int64_t window_right,
                double sm_scale,
                torch::Tensor& output,
                torch::Tensor& output_lse) {
  CHECK(scheduler_metadata.defined() && scheduler_metadata.numel() >= 3)
      << "fa3_decode: scheduler_metadata must be precomputed (size >= 3*B)";
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK(page_table.defined() && page_table.scalar_type() == torch::kInt32);

  const std::string uri = fa3_fwd_uri();
  MusaTvmffiStreamGuard stream_guard(query.device());

  // Slice the flat scheduler_metadata [B*4] into its 3 consumed stripes.
  // Layout matches mate's _fmha_get_metadata: stripes 0..3 hold
  //   [num_splits_dynamic | batch_table | num_m_blocks | num_nheads_in_l2]
  // and fmha_fwd consumes the first three.
  const int64_t b = scheduler_metadata.numel() / 4;
  CHECK_GT(b, 0)
      << "fa3_decode: scheduler_metadata size must be 4*batch_size";
  auto num_splits_dynamic = scheduler_metadata.slice(0, 0, b);
  auto batch_table = scheduler_metadata.slice(0, b, 2 * b);
  auto num_m_blocks = scheduler_metadata.slice(0, 2 * b, 3 * b);

  // 39 positional args. See generated source comment above for ordering.
  get_function(uri, uri)(
      to_ffi_tensor(query),                       // [0]
      to_ffi_tensor(k_cache),                     // [1]
      to_ffi_tensor(v_cache),                     // [2]
      none_tensor(),                              // [3]  k_new
      none_tensor(),                              // [4]  v_new
      none_tensor(),                              // [5]  q_v
      to_ffi_tensor(cu_seqlens_q),                // [6]  cu_seqlens_q
      none_tensor(),                              // [7]  cu_seqlens_k
      none_tensor(),                              // [8]  cu_seqlens_k_new
      none_tensor(),                              // [9]  seqused_q
      to_ffi_tensor(seqused_k),                   // [10] seqused_k
      ffi::Optional<int64_t>(max_seqlen_q),       // [11] max_seqlen_q
      none_int(),                                 // [12] max_seqlen_kv
      to_ffi_tensor(page_table),                  // [13] page_table
      none_tensor(),                              // [14] kv_batch_idx
      none_tensor(),                              // [15] leftpad_k
      none_tensor(),                              // [16] rotary_cos
      none_tensor(),                              // [17] rotary_sin
      none_tensor(),                              // [18] seqlens_rotary
      none_tensor(),                              // [19] q_descale
      none_tensor(),                              // [20] k_descale
      none_tensor(),                              // [21] v_descale
      sm_scale,                                   // [22] softmax_scale
      /*is_causal=*/true,                         // [23]
      window_left,                                // [24] window_size_left
      window_right,                               // [25] window_size_right
      /*attention_chunk=*/static_cast<int64_t>(0),  // [26]
      /*softcap=*/0.0,                            // [27]
      /*mp_margin=*/static_cast<int64_t>(0),      // [28]
      /*num_splits=*/static_cast<int64_t>(0),     // [29]
      to_ffi_tensor(num_splits_dynamic),          // [30]
      to_ffi_tensor(batch_table),                 // [31]
      to_ffi_tensor(num_m_blocks),                // [32]
      none_tensor(),                              // [33] learnable_sink
      to_ffi_tensor(output),                      // [34] out
      to_ffi_tensor(output_lse),                  // [35] lse
      /*cp_world_size=*/static_cast<int64_t>(1),  // [36]
      /*cp_rank=*/static_cast<int64_t>(0),        // [37]
      none_tensor()                               // [38] cp_tot_seqused_k
  );
}

}  // namespace xllm::kernel::cuda
