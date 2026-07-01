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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/cuda.h>

#include <cstdint>

#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {

namespace {

// kernel(row, head_group): write Q[hq], K[hq], V[hq], Z[hq], B[hq], A[hq]
// from a single fused row, mirroring sglang's qkvzba_contiguous.
//
// Layout (per row of `fused`):
//   [Q[0..num_heads_qk*head_qk)               -> mixed_qkv [Q region]
//    K[0..num_heads_qk*head_qk)               -> mixed_qkv [K region]
//    V[0..num_heads_v*head_v)                 -> mixed_qkv [V region]
//    Z[0..num_heads_v*head_v)                 -> z         [reshape (M, num_heads_v, head_v)]
//    B[0..num_heads_v)                        -> b
//    A[0..num_heads_v)]                       -> a
//
// Mapping for head group `hq`:
//   q_off = hq * head_qk + d                    (d in [0, head_qk))
//   k_off = total_q + hq * head_qk + d
//   v_off = total_q + total_k + hq * v_group_dim + d   (d in [0, v_group_dim))
//   z_off = qkv_dim + hq * v_group_dim + d             (same d)
//   ba_b_off = qkv_dim + z_dim + hq * v_per_group + i  (i in [0, v_per_group))
//   ba_a_off = qkv_dim + z_dim + num_heads_v + hq * v_per_group + i
//
//   z output index: z[row, hq * v_per_group + d/head_v, d%head_v]
//                = z_flat[row, hq * v_group_dim + d]            (same d)
template <typename T>
__global__ void qkvzba_split_contiguous_kernel(
    const T* __restrict__ fused,
    T* __restrict__ mixed_qkv,
    T* __restrict__ z,
    T* __restrict__ b,
    T* __restrict__ a,
    int64_t fused_row_stride,
    int64_t mixed_qkv_row_stride,
    int64_t z_row_stride,
    int64_t b_row_stride,
    int64_t a_row_stride,
    int num_heads_qk,
    int num_heads_v,
    int head_qk,
    int head_v,
    int v_per_group,
    int v_group_dim,
    int total_q,
    int total_k,
    int qkv_dim,
    int z_dim) {
  const int row = blockIdx.x;
  const int hq = blockIdx.y;
  const int tid = threadIdx.x;
  const int block_threads = blockDim.x;

  const T* __restrict__ fused_row = fused + row * fused_row_stride;
  T* __restrict__ qkv_row = mixed_qkv + row * mixed_qkv_row_stride;
  T* __restrict__ z_row = z + row * z_row_stride;
  T* __restrict__ b_row = b + row * b_row_stride;
  T* __restrict__ a_row = a + row * a_row_stride;

  for (int d = tid; d < head_qk; d += block_threads) {
    const int q_off = hq * head_qk + d;
    const int k_off = total_q + q_off;
    qkv_row[q_off] = fused_row[q_off];
    qkv_row[k_off] = fused_row[k_off];
  }

  const int v_block_base = total_q + total_k + hq * v_group_dim;
  const int z_block_base = qkv_dim + hq * v_group_dim;
  const int z_out_base = hq * v_group_dim;
  for (int d = tid; d < v_group_dim; d += block_threads) {
    qkv_row[v_block_base + d] = fused_row[v_block_base + d];
    z_row[z_out_base + d] = fused_row[z_block_base + d];
  }

  const int b_in_base = qkv_dim + z_dim + hq * v_per_group;
  const int a_in_base = b_in_base + num_heads_v;
  const int ba_out_base = hq * v_per_group;
  for (int i = tid; i < v_per_group; i += block_threads) {
    b_row[ba_out_base + i] = fused_row[b_in_base + i];
    a_row[ba_out_base + i] = fused_row[a_in_base + i];
  }
}

}  // namespace

// Host wrapper: launch the qkvzba_split kernel that scatters `fused` into the
// four pre-allocated output tensors. All output tensors must already exist
// (callers maintain persistent buffers; this function does NOT allocate).
//
// Tensor expectations:
//   fused       : [M, qkv_dim + z_dim + 2*num_heads_v], contiguous along last
//                 dim (any leading stride permitted).
//   mixed_qkv   : [M, qkv_dim], contiguous along last dim.
//   z           : [M, num_heads_v, head_v] OR [M, num_heads_v * head_v],
//                 contiguous along the last logical dim (we treat it as
//                 [M, z_dim] internally; the kernel writes the same flat
//                 layout that a sglang-style view([M, num_heads_v, head_v])
//                 maps onto).
//   b, a        : [M, num_heads_v], contiguous along last dim.
//
// All tensors must share dtype (BF16 / FP16 / FP32) and reside on the same
// MUSA device. This is the small-batch variant only; for M >= 1024 the
// sglang sources select a row/vec specialization that is not needed for
// xLLM's decode buckets (M <= max_seqs_per_batch).
void gdn_fused_qkvzba_split_contiguous(torch::Tensor fused,
                                       torch::Tensor mixed_qkv,
                                       torch::Tensor z,
                                       torch::Tensor b,
                                       torch::Tensor a,
                                       int64_t num_heads_qk,
                                       int64_t num_heads_v,
                                       int64_t head_qk,
                                       int64_t head_v) {
  // MUSA-as-CUDA builds: tensors flowing in from layer output buffers (e.g.
  // ColumnParallelLinear's persistent output_buf_) are allocated as
  // PrivateUse1 by torch_musa, while other kernels' outputs may be reported
  // as kCUDA. Accept either to match the device-detection idiom used in
  // musa_tvmffi_stream.h.
  CHECK(fused.is_cuda() || fused.device().is_privateuseone())
      << "fused must be on CUDA / MUSA device, got " << fused.device();
  CHECK_EQ(fused.scalar_type(), mixed_qkv.scalar_type());
  CHECK_EQ(fused.scalar_type(), z.scalar_type());
  CHECK_EQ(fused.scalar_type(), b.scalar_type());
  CHECK_EQ(fused.scalar_type(), a.scalar_type());
  CHECK_EQ(fused.dim(), 2);
  CHECK_EQ(mixed_qkv.dim(), 2);
  CHECK_EQ(b.dim(), 2);
  CHECK_EQ(a.dim(), 2);
  CHECK(z.dim() == 2 || z.dim() == 3)
      << "z must be 2D [M, num_v_heads*head_v] or 3D [M, num_v_heads, head_v]";
  CHECK(fused.stride(-1) == 1);
  CHECK(mixed_qkv.stride(-1) == 1);
  CHECK(b.stride(-1) == 1);
  CHECK(a.stride(-1) == 1);
  CHECK(z.is_contiguous())
      << "z must be contiguous (kernel writes via flat indexing)";

  const int64_t M = fused.size(0);
  TORCH_CHECK(num_heads_qk > 0 && num_heads_v > 0 && head_qk > 0 && head_v > 0,
              "gdn_fused_qkvzba_split_contiguous: head dims must be positive");
  TORCH_CHECK(num_heads_v % num_heads_qk == 0,
              "gdn_fused_qkvzba_split_contiguous: num_heads_v (",
              num_heads_v,
              ") must be a multiple of num_heads_qk (",
              num_heads_qk,
              ")");
  const int64_t v_per_group = num_heads_v / num_heads_qk;
  const int64_t v_group_dim = v_per_group * head_v;
  const int64_t total_q = num_heads_qk * head_qk;
  const int64_t total_k = total_q;
  const int64_t z_dim = num_heads_v * head_v;
  const int64_t qkv_dim = total_q + total_k + num_heads_v * head_v;
  const int64_t fused_dim_expected = qkv_dim + z_dim + 2 * num_heads_v;

  TORCH_CHECK(fused.size(1) == fused_dim_expected,
              "gdn_fused_qkvzba_split_contiguous: fused last dim mismatch, got ",
              fused.size(1),
              " expected ",
              fused_dim_expected);
  TORCH_CHECK(mixed_qkv.size(0) >= M && mixed_qkv.size(1) == qkv_dim,
              "gdn_fused_qkvzba_split_contiguous: mixed_qkv shape mismatch");
  TORCH_CHECK(b.size(0) >= M && b.size(1) == num_heads_v,
              "gdn_fused_qkvzba_split_contiguous: b shape mismatch");
  TORCH_CHECK(a.size(0) >= M && a.size(1) == num_heads_v,
              "gdn_fused_qkvzba_split_contiguous: a shape mismatch");
  TORCH_CHECK(z.numel() / M == z_dim,
              "gdn_fused_qkvzba_split_contiguous: z last-dim mismatch");

  const int64_t fused_row_stride = fused.stride(0);
  const int64_t mixed_qkv_row_stride = mixed_qkv.stride(0);
  const int64_t z_row_stride = z_dim;  // contiguous, see CHECK above
  const int64_t b_row_stride = b.stride(0);
  const int64_t a_row_stride = a.stride(0);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(fused));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Same launch shape as sglang's TileLang qkvzba_contiguous (target="musa",
  // threads=128).  For decode buckets (M <= 16) this means at most 16 *
  // num_heads_qk blocks, each writing ~2*head_qk + 2*v_group_dim + 2*v_per_group
  // contiguous elements -- a single L1 cache line worth of work per warp.
  const dim3 grid(static_cast<unsigned int>(M),
                  static_cast<unsigned int>(num_heads_qk),
                  1u);
  constexpr unsigned int kBlockThreads = 128;

  AT_DISPATCH_SWITCH(
      fused.scalar_type(),
      "gdn_fused_qkvzba_split_contiguous",
      AT_DISPATCH_CASE(at::ScalarType::Half,
                       [&] {
                         qkvzba_split_contiguous_kernel<__half>
                             <<<grid, kBlockThreads, 0, stream>>>(
                                 reinterpret_cast<const __half*>(
                                     fused.data_ptr<c10::Half>()),
                                 reinterpret_cast<__half*>(
                                     mixed_qkv.data_ptr<c10::Half>()),
                                 reinterpret_cast<__half*>(
                                     z.data_ptr<c10::Half>()),
                                 reinterpret_cast<__half*>(
                                     b.data_ptr<c10::Half>()),
                                 reinterpret_cast<__half*>(
                                     a.data_ptr<c10::Half>()),
                                 fused_row_stride,
                                 mixed_qkv_row_stride,
                                 z_row_stride,
                                 b_row_stride,
                                 a_row_stride,
                                 static_cast<int>(num_heads_qk),
                                 static_cast<int>(num_heads_v),
                                 static_cast<int>(head_qk),
                                 static_cast<int>(head_v),
                                 static_cast<int>(v_per_group),
                                 static_cast<int>(v_group_dim),
                                 static_cast<int>(total_q),
                                 static_cast<int>(total_k),
                                 static_cast<int>(qkv_dim),
                                 static_cast<int>(z_dim));
                       })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
                       [&] {
                         qkvzba_split_contiguous_kernel<__nv_bfloat16>
                             <<<grid, kBlockThreads, 0, stream>>>(
                                 reinterpret_cast<const __nv_bfloat16*>(
                                     fused.data_ptr<c10::BFloat16>()),
                                 reinterpret_cast<__nv_bfloat16*>(
                                     mixed_qkv.data_ptr<c10::BFloat16>()),
                                 reinterpret_cast<__nv_bfloat16*>(
                                     z.data_ptr<c10::BFloat16>()),
                                 reinterpret_cast<__nv_bfloat16*>(
                                     b.data_ptr<c10::BFloat16>()),
                                 reinterpret_cast<__nv_bfloat16*>(
                                     a.data_ptr<c10::BFloat16>()),
                                 fused_row_stride,
                                 mixed_qkv_row_stride,
                                 z_row_stride,
                                 b_row_stride,
                                 a_row_stride,
                                 static_cast<int>(num_heads_qk),
                                 static_cast<int>(num_heads_v),
                                 static_cast<int>(head_qk),
                                 static_cast<int>(head_v),
                                 static_cast<int>(v_per_group),
                                 static_cast<int>(v_group_dim),
                                 static_cast<int>(total_q),
                                 static_cast<int>(total_k),
                                 static_cast<int>(qkv_dim),
                                 static_cast<int>(z_dim));
                       })
      AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
        qkvzba_split_contiguous_kernel<float>
            <<<grid, kBlockThreads, 0, stream>>>(fused.data_ptr<float>(),
                                                 mixed_qkv.data_ptr<float>(),
                                                 z.data_ptr<float>(),
                                                 b.data_ptr<float>(),
                                                 a.data_ptr<float>(),
                                                 fused_row_stride,
                                                 mixed_qkv_row_stride,
                                                 z_row_stride,
                                                 b_row_stride,
                                                 a_row_stride,
                                                 static_cast<int>(num_heads_qk),
                                                 static_cast<int>(num_heads_v),
                                                 static_cast<int>(head_qk),
                                                 static_cast<int>(head_v),
                                                 static_cast<int>(v_per_group),
                                                 static_cast<int>(v_group_dim),
                                                 static_cast<int>(total_q),
                                                 static_cast<int>(total_k),
                                                 static_cast<int>(qkv_dim),
                                                 static_cast<int>(z_dim));
      }));
}

}  // namespace xllm::kernel::cuda
