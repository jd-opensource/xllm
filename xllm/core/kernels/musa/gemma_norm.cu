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

// Gemma-style RMS norm kernels for the USE_CUDA path (also covers MUSA-as-CUDA
// builds via mcc -x musa + libMusaMapping.so).
//
// The kernels here are functionally equivalent to the standard rms_norm in
// norm.cu, but apply the `(weight + 1.0)` scaling that the Gemma variant
// requires. Crucially, the `+1.0` is fused into the kernel so callers do NOT
// need to materialize `(1.0 + weight)` as a torch::Tensor on the host -- that
// host-side allocation is the same thing that broke MUSA graph capture in
// `cuda_fallback::gemma_rms_norm` (9 intermediate allocations per call).
//
// Algorithm + vectorization + MUSA arch-310 asm hints are ported as a
// near 1:1 translation of sglang's
// `python/sglang/srt/hardware_backend/musa/jit_kernel/csrc/norm/rmsnorm.mu`.
// Specifically:
//   * `Vec8<T>::load_byp_slc` -> `LSU.LD.B128 ... slc=byp, chrnt=l2_l3` for the
//      one-shot reads in `fused_add_rmsnorm_vec8_kernel`'s first pass (input +
//      residual are read then immediately overwritten -- L2/L3 caching is
//      pointless).
//   * `fast_rsqrt` -> `__frsqrt_rn` + 1 Newton-Raphson iteration on arch 310.
//   * `block_sum` -> `__syncthreads_lm` lightweight barrier.
//   * Small-hidden register specializations for `hidden==1024` (WARPS=4) and
//      `hidden==2048` (WARPS=8) when `rows<=16` -- keeps the per-thread
//      Float8 in registers rather than smem.
//
// Why graph-capture-safe: zero intermediate tensor allocations, one kernel
// launch per call. The caller (Qwen3NextRMSNormImpl) owns a persistent
// output buffer that gets pre-allocated during warmup; during capture the
// kernel just writes into that buffer.

#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/cuda.h>

#include <cstdint>

#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {

namespace {

// ---------------------------------------------------------------------------
// Vec8 / Float8: 16- and 32-byte aligned containers for 8-wide vector load
// stores. Reading/writing them as `*(Vec8<T>*)(ptr + col)` issues a single
// 128-bit ld.global / st.global instruction on the kernel hot loop.
// ---------------------------------------------------------------------------

template <typename T>
struct __align__(16) Vec8Storage {
  T elem[8];
};

struct __align__(32) Float8Storage {
  float elem[8];
};

template <typename T>
struct __align__(16) Vec8 {
  union {
    Vec8Storage<T> storage;
    T elem[8];
  } val;

  __device__ __forceinline__ Vec8() {}

  // Standard 128-bit aligned load.
  template <typename Offset>
  static __device__ __forceinline__ Vec8 load(const T* ptr, Offset idx) {
    return *(const Vec8*)(ptr + idx);
  }

  // Bypass-L2/L3 load. Used for one-shot reads where the data is consumed
  // immediately and never re-read (e.g., the `input` + `residual` reads in
  // `fused_add_rmsnorm_vec8_kernel`'s first pass, which are immediately
  // written back as `residual_out` in the second pass). Avoiding cache
  // pollution here improves bandwidth for the rest of the kernel.
  //
  // The MUSA arch-310 inline asm requires a 128-bit destination register;
  // older mcc versions reject `"=R"(struct_type)` constraints, so we route
  // through `uint4` (the canonical 4xu32 packed type) and then bit-cast.
  // The `__align__(16)` on Vec8 ensures the underlying storage is aligned,
  // and the bit-cast compiles to a no-op register move.
  // On non-arch-310 targets this falls back to a plain aligned load.
  template <typename Offset>
  static __device__ __forceinline__ Vec8 load_byp_slc(const T* ptr,
                                                       Offset idx) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
    uint4 raw;
    const T* addr = ptr + idx;
    asm volatile(
        "LSU.LD.B128 %0, %1, _, 16, 1, 1, inner_persist=0, outer_persist=2, "
        "chrnt=l2_l3, slc=byp, persist=0, stride_add_first=0"
        : "=R"(raw)
        : "R"(addr));
    Vec8 dst;
    *reinterpret_cast<uint4*>(&dst) = raw;
    return dst;
#else
    return *(const Vec8*)(ptr + idx);
#endif
  }
};

struct __align__(32) Float8 {
  union {
    Float8Storage storage;
    float elem[8];
  } val;

  __device__ __forceinline__ Float8() {}
};

// ---------------------------------------------------------------------------
// dtype helpers (fp16/bf16/fp32 → float, and back).
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float gemma_to_float(T value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else {
    return static_cast<float>(value);
  }
}

template <typename T>
__device__ __forceinline__ T gemma_from_float(float value) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16_rn(value);
  } else {
    return static_cast<T>(value);
  }
}

// rsqrt with optional Newton refinement step on MUSA arch 310. The hardware
// `__frsqrt_rn` is a low-precision instruction (a few ulp of error); one
// Newton-Raphson iteration brings it to within 1 ulp without measurably
// hurting throughput because the cost is two fused multiply-adds. On other
// targets we use `rsqrtf` which is already correctly-rounded.
__device__ __forceinline__ float fast_rsqrt(float value) {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  const float half_value = 0.5f * value;
  float y = __frsqrt_rn(value);
  y = y * (1.5f - half_value * y * y);
  return y;
#else
  return rsqrtf(value);
#endif
}

// Lightweight barrier. MUSA arch 310 has `__syncthreads_lm` (locally-mapped
// metadata sync) which is cheaper than the full `__syncthreads`. Fall back
// to the standard barrier on non-MUSA / pre-310 targets.
__device__ __forceinline__ void block_sync() {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  __syncthreads_lm();
#else
  __syncthreads();
#endif
}

// ---------------------------------------------------------------------------
// Block-wide reduction (sum) with shared-memory warp scratch.
//   - block_sum is the generic variant; the kernel passes blockDim.x.
//   - block_sum_4warps / block_sum_8warps are explicit specializations used
//     by the small-hidden register kernels (4 or 8 warps fixed).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  block_sync();

  value = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  block_sync();
  return warp_sums[0];
}

// Specialized block-wide reduction for blocks of exactly 8 warps (256
// threads). Used by the `H=2048` register-kernel specialization. Saves the
// runtime `num_warps` computation and the conditional load mask.
__device__ __forceinline__ float block_sum_8warps(float value,
                                                  float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  block_sync();

  value = lane < 8 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  block_sync();
  return warp_sums[0];
}

// Specialized block-wide reduction for blocks of exactly 4 warps (128
// threads). Used by the `H=1024` register-kernel specialization.
__device__ __forceinline__ float block_sum_4warps(float value,
                                                  float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  block_sync();

  value = lane < 4 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  block_sync();
  return warp_sums[0];
}

// ---------------------------------------------------------------------------
// Main vec8 RMS norm kernel.
//
// Template params:
//   T      = element dtype (half, bfloat16, float).
//   GEMMA  = if true, the per-element output is `x * scale * (weight + 1.0f)`.
//            If false, the standard `x * scale * weight`. Set true for the
//            gemma launchers; false is left for completeness (and lets us
//            share this kernel with non-gemma launches if we ever wire it up).
//   CACHE  = if true, the first pass writes the fp32-promoted input values to
//            shared memory so the second pass can read them back without a
//            redundant gmem load. Use for hidden sizes that fit in smem.
// ---------------------------------------------------------------------------

template <typename T, bool GEMMA, bool CACHE>
__global__ void __launch_bounds__(1024, 1)
    rmsnorm_vec8_kernel(const T* __restrict__ input,
                        const T* __restrict__ weight,
                        T* __restrict__ out,
                        int rows,
                        int hidden,
                        int input_outer_dim,
                        int64_t input_outer_stride,
                        int64_t input_row_stride,
                        int64_t out_row_stride,
                        float inv_hidden,
                        float eps) {
  constexpr int kVec = 8;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + (CACHE ? hidden : 0);

  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / kVec;
  // 3D stride-aware indexing. For a 3D non-contiguous view [N, H, D] with
  // strides [s0, s1, 1] (e.g. Qwen3.5 GDN q/k slices of fused QKV), the
  // flat-row r = i*H + j must land at i*s0 + j*s1, not r*s1. For 2D inputs
  // the host wrapper passes `input_outer_dim = rows` (so `row /
  // input_outer_dim == 0` for all rows in the grid) and
  // `input_outer_stride = 0`, which collapses the formula to the original
  // `row * input_row_stride`.
  const int64_t input_base =
      static_cast<int64_t>(row / input_outer_dim) * input_outer_stride +
      static_cast<int64_t>(row % input_outer_dim) * input_row_stride;
  const int64_t out_base = static_cast<int64_t>(row) * out_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count;
       vec_idx += static_cast<int>(blockDim.x)) {
    const int col = vec_idx * kVec;
    Vec8<T> x = Vec8<T>::load(input + input_base, col);
    Float8 x_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value = gemma_to_float<T>(x.val.elem[i]);
      sum += value * value;
      x_float.val.elem[i] = value;
    }
    if constexpr (CACHE) {
      *(Float8*)(cached + col) = x_float;
    }
  }

  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count;
       vec_idx += static_cast<int>(blockDim.x)) {
    const int col = vec_idx * kVec;
    Float8 x_float;
    if constexpr (CACHE) {
      x_float = *(Float8*)(cached + col);
    } else {
      Vec8<T> x = Vec8<T>::load(input + input_base, col);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        x_float.val.elem[i] = gemma_to_float<T>(x.val.elem[i]);
      }
    }
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value =
          gemma_to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] =
          gemma_from_float<T>(x_float.val.elem[i] * scale * weight_value);
    }
    *(Vec8<T>*)(out + out_base + col) = dst;
  }
}

// Small-hidden register-resident kernel. One Vec8 per thread covers the full
// row in a single pass; the FP32-promoted view sits in registers instead of
// shared memory, eliminating the smem load/store traffic the general kernel
// pays in pass 2. Used only for `rows<=16 && hidden in {1024, 2048}` where
// the register pressure is acceptable.
template <typename T, bool GEMMA, int H, int WARPS>
__global__ void __launch_bounds__(256, 1)
    rmsnorm_small_h_one_vec_register_kernel(const T* __restrict__ input,
                                            const T* __restrict__ weight,
                                            T* __restrict__ out,
                                            int rows,
                                            int hidden,
                                            int input_outer_dim,
                                            int64_t input_outer_stride,
                                            int64_t input_row_stride,
                                            int64_t out_row_stride,
                                            float inv_hidden,
                                            float eps) {
  constexpr int kVec = 8;
  extern __shared__ float warp_sums[];

  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int col = tid * kVec;
  // See `rmsnorm_vec8_kernel` for the stride-aware indexing rationale.
  const int64_t input_base =
      static_cast<int64_t>(row / input_outer_dim) * input_outer_stride +
      static_cast<int64_t>(row % input_outer_dim) * input_row_stride;
  const int64_t out_base = static_cast<int64_t>(row) * out_row_stride;
  float sum = 0.0f;
  Float8 x_float;

  Vec8<T> x = Vec8<T>::load(input + input_base, col);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float value = gemma_to_float<T>(x.val.elem[i]);
    sum += value * value;
    x_float.val.elem[i] = value;
  }

  if constexpr (WARPS == 4) {
    sum = block_sum_4warps(sum, warp_sums);
  } else {
    sum = block_sum_8warps(sum, warp_sums);
  }

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  Vec8<T> w = Vec8<T>::load(weight, col);
  Vec8<T> dst;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float weight_value =
        gemma_to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
    dst.val.elem[i] =
        gemma_from_float<T>(x_float.val.elem[i] * scale * weight_value);
  }
  *(Vec8<T>*)(out + out_base + col) = dst;
}

// Scalar fallback for hidden sizes that aren't multiples of 8.
template <typename T, bool GEMMA>
__global__ void rmsnorm_scalar_kernel(const T* __restrict__ input,
                                       const T* __restrict__ weight,
                                       T* __restrict__ out,
                                       int rows,
                                       int hidden,
                                       int input_outer_dim,
                                       int64_t input_outer_stride,
                                       int64_t input_row_stride,
                                       int64_t out_row_stride,
                                       float inv_hidden,
                                       float eps) {
  extern __shared__ float warp_sums[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  // See `rmsnorm_vec8_kernel` for the stride-aware indexing rationale.
  const int64_t input_base =
      static_cast<int64_t>(row / input_outer_dim) * input_outer_stride +
      static_cast<int64_t>(row % input_outer_dim) * input_row_stride;
  const int64_t out_base = static_cast<int64_t>(row) * out_row_stride;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += static_cast<int>(blockDim.x)) {
    const float value = gemma_to_float<T>(input[input_base + col]);
    sum += value * value;
  }
  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int col = tid; col < hidden; col += static_cast<int>(blockDim.x)) {
    const float weight_value =
        gemma_to_float<T>(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    out[out_base + col] = gemma_from_float<T>(
        gemma_to_float<T>(input[input_base + col]) * scale * weight_value);
  }
}

// Fused-add variant: writes the post-add value into `residual` (in-place),
// then re-uses it for the rmsnorm step that writes back into `input`. Mirrors
// the no-residual layout but with one extra Vec8 load per row in the first
// pass and an extra Vec8 store of `residual_out`.
template <typename T, bool GEMMA, bool CACHE>
__global__ void __launch_bounds__(1024, 1)
    fused_add_rmsnorm_vec8_kernel(T* __restrict__ input,
                                  T* __restrict__ residual,
                                  const T* __restrict__ weight,
                                  int rows,
                                  int hidden,
                                  int64_t input_row_stride,
                                  int64_t residual_row_stride,
                                  float inv_hidden,
                                  float eps) {
  constexpr int kVec = 8;
  extern __shared__ __align__(16) float smem[];
  float* cached = smem;
  float* warp_sums = smem + (CACHE ? hidden : 0);

  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / kVec;
  const int64_t input_base = static_cast<int64_t>(row) * input_row_stride;
  const int64_t residual_base =
      static_cast<int64_t>(row) * residual_row_stride;
  float sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count;
       vec_idx += static_cast<int>(blockDim.x)) {
    const int col = vec_idx * kVec;
    // `input` and `residual` are each read EXACTLY ONCE before being
    // overwritten (residual gets the post-add value, input gets the norm
    // output in pass 2). Bypass L2/L3 caching for these reads so they don't
    // evict the weight tensor or the cached fp32 buffer.
    Vec8<T> x = Vec8<T>::load_byp_slc(input + input_base, col);
    Vec8<T> r = Vec8<T>::load_byp_slc(residual + residual_base, col);
    Vec8<T> residual_out;
    Float8 sum_float;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float value =
          gemma_to_float<T>(x.val.elem[i]) + gemma_to_float<T>(r.val.elem[i]);
      sum += value * value;
      residual_out.val.elem[i] = gemma_from_float<T>(value);
      sum_float.val.elem[i] = value;
    }
    *(Vec8<T>*)(residual + residual_base + col) = residual_out;
    if constexpr (CACHE) {
      *(Float8*)(cached + col) = sum_float;
    }
  }

  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int vec_idx = tid; vec_idx < vec_count;
       vec_idx += static_cast<int>(blockDim.x)) {
    const int col = vec_idx * kVec;
    Float8 sum_float;
    if constexpr (CACHE) {
      sum_float = *(Float8*)(cached + col);
    } else {
      Vec8<T> r = Vec8<T>::load(residual + residual_base, col);
#pragma unroll
      for (int i = 0; i < kVec; ++i) {
        sum_float.val.elem[i] = gemma_to_float<T>(r.val.elem[i]);
      }
    }
    Vec8<T> w = Vec8<T>::load(weight, col);
    Vec8<T> dst;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      const float weight_value =
          gemma_to_float<T>(w.val.elem[i]) + (GEMMA ? 1.0f : 0.0f);
      dst.val.elem[i] =
          gemma_from_float<T>(sum_float.val.elem[i] * scale * weight_value);
    }
    *(Vec8<T>*)(input + input_base + col) = dst;
  }
}

template <typename T, bool GEMMA>
__global__ void fused_add_rmsnorm_scalar_kernel(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    int rows,
    int hidden,
    int64_t input_row_stride,
    int64_t residual_row_stride,
    float inv_hidden,
    float eps) {
  extern __shared__ float warp_sums[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t input_base = static_cast<int64_t>(row) * input_row_stride;
  const int64_t residual_base =
      static_cast<int64_t>(row) * residual_row_stride;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += static_cast<int>(blockDim.x)) {
    const float value =
        gemma_to_float<T>(input[input_base + col]) +
        gemma_to_float<T>(residual[residual_base + col]);
    residual[residual_base + col] = gemma_from_float<T>(value);
    sum += value * value;
  }
  sum = block_sum(sum, warp_sums);

  const float scale = fast_rsqrt(sum * inv_hidden + eps);
  for (int col = tid; col < hidden; col += static_cast<int>(blockDim.x)) {
    const float weight_value =
        gemma_to_float<T>(weight[col]) + (GEMMA ? 1.0f : 0.0f);
    input[input_base + col] = gemma_from_float<T>(
        gemma_to_float<T>(residual[residual_base + col]) * scale *
        weight_value);
  }
}

// ---------------------------------------------------------------------------
// Host-side dispatch helpers (block sizing, smem sizing).
// ---------------------------------------------------------------------------

inline int vec8_block_threads(int hidden) {
  const int vec_count = hidden / 8;
  const int rounded = ((vec_count + 31) / 32) * 32;
  return rounded < 1024 ? rounded : 1024;
}

inline int rmsnorm_block_threads(int rows, int hidden) {
  if (hidden <= 512) {
    return 64;
  }
  if (hidden <= 4096) {
    if (rows <= 16) {
      const int threads = vec8_block_threads(hidden);
      return threads < 512 ? threads : 512;
    }
    if (rows <= 256) {
      const int threads = vec8_block_threads(hidden);
      return threads < 256 ? threads : 256;
    }
    return 128;
  }
  if (hidden <= 8192) {
    const int threads = vec8_block_threads(hidden);
    return threads < 512 ? threads : 512;
  }
  const int threads = vec8_block_threads(hidden);
  return threads < 896 ? threads : 896;
}

inline int fused_block_threads(int hidden) {
  return vec8_block_threads(hidden);
}

inline int cached_vec8_shared_bytes(int hidden,
                                    int block_threads,
                                    int cache_hidden_limit) {
  const int reduce_floats = (block_threads + 31) / 32;
  const int cached_floats = hidden <= cache_hidden_limit ? hidden : 0;
  return (cached_floats + reduce_floats) * static_cast<int>(sizeof(float));
}

// scalar_t (torch fp16/bf16/fp32) → CUDA dtype shim used by the templates
// above. We dispatch via the torch scalar type and call into the matching
// kernel instantiation.
template <typename T>
void launch_rmsnorm_gemma(const T* input_ptr,
                          const T* weight_ptr,
                          T* out_ptr,
                          int rows,
                          int hidden,
                          int input_outer_dim,
                          int64_t input_outer_stride,
                          int64_t input_row_stride,
                          int64_t out_row_stride,
                          float inv_hidden,
                          float eps,
                          cudaStream_t stream) {
  if ((hidden % 8) == 0 && hidden <= 32768) {
    // Small-hidden register-resident specializations. Each thread holds one
    // Vec8 row chunk in registers across both passes (no smem cached
    // buffer). Layout matches sglang: 128 threads (4 warps) for hidden=1024
    // and 256 threads (8 warps) for hidden=2048. Only `rows<=16` because
    // register pressure dominates throughput once we have more concurrent
    // rows than the SM can pipeline.
    if (rows <= 16 && hidden == 1024) {
      constexpr int threads = 128;
      constexpr int smem_bytes = 4 * static_cast<int>(sizeof(float));
      rmsnorm_small_h_one_vec_register_kernel<T,
                                              /*GEMMA=*/true,
                                              /*H=*/1024,
                                              /*WARPS=*/4>
          <<<rows, threads, smem_bytes, stream>>>(input_ptr,
                                                  weight_ptr,
                                                  out_ptr,
                                                  rows,
                                                  hidden,
                                                  input_outer_dim,
                                                  input_outer_stride,
                                                  input_row_stride,
                                                  out_row_stride,
                                                  inv_hidden,
                                                  eps);
      return;
    }
    if (rows <= 16 && hidden == 2048) {
      constexpr int threads = 256;
      constexpr int smem_bytes = 8 * static_cast<int>(sizeof(float));
      rmsnorm_small_h_one_vec_register_kernel<T,
                                              /*GEMMA=*/true,
                                              /*H=*/2048,
                                              /*WARPS=*/8>
          <<<rows, threads, smem_bytes, stream>>>(input_ptr,
                                                  weight_ptr,
                                                  out_ptr,
                                                  rows,
                                                  hidden,
                                                  input_outer_dim,
                                                  input_outer_stride,
                                                  input_row_stride,
                                                  out_row_stride,
                                                  inv_hidden,
                                                  eps);
      return;
    }
    const int threads = rmsnorm_block_threads(rows, hidden);
    if (hidden <= 8192) {
      const int smem = cached_vec8_shared_bytes(hidden, threads, 8192);
      rmsnorm_vec8_kernel<T, /*GEMMA=*/true, /*CACHE=*/true>
          <<<rows, threads, smem, stream>>>(input_ptr,
                                            weight_ptr,
                                            out_ptr,
                                            rows,
                                            hidden,
                                            input_outer_dim,
                                            input_outer_stride,
                                            input_row_stride,
                                            out_row_stride,
                                            inv_hidden,
                                            eps);
    } else {
      const int smem = cached_vec8_shared_bytes(hidden, threads, 8192);
      rmsnorm_vec8_kernel<T, /*GEMMA=*/true, /*CACHE=*/false>
          <<<rows, threads, smem, stream>>>(input_ptr,
                                            weight_ptr,
                                            out_ptr,
                                            rows,
                                            hidden,
                                            input_outer_dim,
                                            input_outer_stride,
                                            input_row_stride,
                                            out_row_stride,
                                            inv_hidden,
                                            eps);
    }
  } else {
    constexpr int threads = 256;
    constexpr int smem =
        ((threads + 31) / 32) * static_cast<int>(sizeof(float));
    rmsnorm_scalar_kernel<T, /*GEMMA=*/true>
        <<<rows, threads, smem, stream>>>(input_ptr,
                                          weight_ptr,
                                          out_ptr,
                                          rows,
                                          hidden,
                                          input_outer_dim,
                                          input_outer_stride,
                                          input_row_stride,
                                          out_row_stride,
                                          inv_hidden,
                                          eps);
  }
}

template <typename T>
void launch_fused_add_rmsnorm_gemma(T* input_ptr,
                                    T* residual_ptr,
                                    const T* weight_ptr,
                                    int rows,
                                    int hidden,
                                    int64_t input_row_stride,
                                    int64_t residual_row_stride,
                                    float inv_hidden,
                                    float eps,
                                    cudaStream_t stream) {
  if ((hidden % 8) == 0 && hidden <= 32768) {
    const int threads = fused_block_threads(hidden);
    const int smem = cached_vec8_shared_bytes(hidden, threads, 32768);
    fused_add_rmsnorm_vec8_kernel<T, /*GEMMA=*/true, /*CACHE=*/true>
        <<<rows, threads, smem, stream>>>(input_ptr,
                                          residual_ptr,
                                          weight_ptr,
                                          rows,
                                          hidden,
                                          input_row_stride,
                                          residual_row_stride,
                                          inv_hidden,
                                          eps);
  } else {
    constexpr int threads = 256;
    constexpr int smem =
        ((threads + 31) / 32) * static_cast<int>(sizeof(float));
    fused_add_rmsnorm_scalar_kernel<T, /*GEMMA=*/true>
        <<<rows, threads, smem, stream>>>(input_ptr,
                                          residual_ptr,
                                          weight_ptr,
                                          rows,
                                          hidden,
                                          input_row_stride,
                                          residual_row_stride,
                                          inv_hidden,
                                          eps);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public host-side API.
// ---------------------------------------------------------------------------

void gemma_rms_norm(torch::Tensor output,
                    torch::Tensor input,
                    torch::Tensor weight,
                    double eps) {
  CHECK(input.scalar_type() == output.scalar_type());
  CHECK(input.scalar_type() == weight.scalar_type());
  CHECK(output.is_contiguous());
  CHECK(weight.is_contiguous());
  CHECK(input.stride(-1) == 1)
      << "gemma_rms_norm requires the last dim to be contiguous "
         "(stride(-1)==1). Got strides=" << input.strides()
      << ", sizes=" << input.sizes();
  CHECK(input.dim() >= 2 && input.dim() <= 3)
      << "gemma_rms_norm supports 2D [rows, hidden] or 3D [outer, mid, "
         "hidden] inputs only. Got sizes=" << input.sizes();

  const int hidden = input.size(-1);
  const int rows = input.numel() / hidden;
  const int64_t input_row_stride = input.stride(-2);
  // 3D stride-aware launch: for non-contiguous views like Qwen3.5 GDN q/k
  // slices of fused QKV (`qkv[:, 0:24, :]` with stride `[14336, 256, 1]`)
  // the kernel cannot use flat `row * stride(-2)` math because that would
  // collapse the outer dim incorrectly. We pass the outer-dim size and the
  // outer-dim stride separately so the kernel can compute
  // `(row / outer_dim) * outer_stride + (row % outer_dim) * row_stride`.
  // For 2D inputs we set `outer_dim = rows` and `outer_stride = 0`, which
  // makes `row / outer_dim == 0` (since blockIdx.x < rows) and the formula
  // degenerates to the original `row * row_stride`. This avoids any host
  // allocation and is therefore safe under MUSA graph capture (torch_musa
  // 2.7.1 forbids `EmptyStridedMUSA -> musaMemMap` during capture).
  const int input_outer_dim =
      input.dim() == 3 ? static_cast<int>(input.size(-2)) : rows;
  const int64_t input_outer_stride =
      input.dim() == 3 ? input.stride(-3) : 0;
  const int64_t out_row_stride = output.stride(-2);
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "gemma_rms_norm",
      AT_DISPATCH_CASE(at::ScalarType::Half,
                       [&] {
                         launch_rmsnorm_gemma<__half>(
                             reinterpret_cast<const __half*>(
                                 input.data_ptr<c10::Half>()),
                             reinterpret_cast<const __half*>(
                                 weight.data_ptr<c10::Half>()),
                             reinterpret_cast<__half*>(
                                 output.data_ptr<c10::Half>()),
                             rows,
                             hidden,
                             input_outer_dim,
                             input_outer_stride,
                             input_row_stride,
                             out_row_stride,
                             inv_hidden,
                             static_cast<float>(eps),
                             stream);
                       })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
                       [&] {
                         launch_rmsnorm_gemma<__nv_bfloat16>(
                             reinterpret_cast<const __nv_bfloat16*>(
                                 input.data_ptr<c10::BFloat16>()),
                             reinterpret_cast<const __nv_bfloat16*>(
                                 weight.data_ptr<c10::BFloat16>()),
                             reinterpret_cast<__nv_bfloat16*>(
                                 output.data_ptr<c10::BFloat16>()),
                             rows,
                             hidden,
                             input_outer_dim,
                             input_outer_stride,
                             input_row_stride,
                             out_row_stride,
                             inv_hidden,
                             static_cast<float>(eps),
                             stream);
                       })
      AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
        launch_rmsnorm_gemma<float>(input.data_ptr<float>(),
                                    weight.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    rows,
                                    hidden,
                                    input_outer_dim,
                                    input_outer_stride,
                                    input_row_stride,
                                    out_row_stride,
                                    inv_hidden,
                                    static_cast<float>(eps),
                                    stream);
      }));
}

void fused_add_gemma_rms_norm(torch::Tensor& input,
                              torch::Tensor& residual,
                              torch::Tensor& weight,
                              double epsilon) {
  CHECK(input.scalar_type() == residual.scalar_type());
  CHECK(input.scalar_type() == weight.scalar_type());
  CHECK(residual.is_contiguous());
  CHECK(weight.is_contiguous());
  // Same flat-row contiguity requirement as `gemma_rms_norm` above. The
  // residual path writes back in-place to both `input` and `residual`, so
  // we cannot silently copy `input` (a caller that holds the same tensor
  // would lose the in-place semantics). We therefore CHECK contiguity
  // instead of attempting to make it contiguous.
  CHECK(input.is_contiguous())
      << "fused_add_gemma_rms_norm requires a contiguous `input` (in-place "
         "write back). Got strides=" << input.strides()
      << ", sizes=" << input.sizes();

  const int hidden = input.size(-1);
  const int64_t input_row_stride = input.stride(-2);
  const int64_t residual_row_stride = residual.stride(-2);
  const int rows = input.numel() / hidden;
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "fused_add_gemma_rms_norm",
      AT_DISPATCH_CASE(at::ScalarType::Half,
                       [&] {
                         launch_fused_add_rmsnorm_gemma<__half>(
                             reinterpret_cast<__half*>(
                                 input.data_ptr<c10::Half>()),
                             reinterpret_cast<__half*>(
                                 residual.data_ptr<c10::Half>()),
                             reinterpret_cast<const __half*>(
                                 weight.data_ptr<c10::Half>()),
                             rows,
                             hidden,
                             input_row_stride,
                             residual_row_stride,
                             inv_hidden,
                             static_cast<float>(epsilon),
                             stream);
                       })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
                       [&] {
                         launch_fused_add_rmsnorm_gemma<__nv_bfloat16>(
                             reinterpret_cast<__nv_bfloat16*>(
                                 input.data_ptr<c10::BFloat16>()),
                             reinterpret_cast<__nv_bfloat16*>(
                                 residual.data_ptr<c10::BFloat16>()),
                             reinterpret_cast<const __nv_bfloat16*>(
                                 weight.data_ptr<c10::BFloat16>()),
                             rows,
                             hidden,
                             input_row_stride,
                             residual_row_stride,
                             inv_hidden,
                             static_cast<float>(epsilon),
                             stream);
                       })
      AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
        launch_fused_add_rmsnorm_gemma<float>(input.data_ptr<float>(),
                                              residual.data_ptr<float>(),
                                              weight.data_ptr<float>(),
                                              rows,
                                              hidden,
                                              input_row_stride,
                                              residual_row_stride,
                                              inv_hidden,
                                              static_cast<float>(epsilon),
                                              stream);
      }));
}

}  // namespace xllm::kernel::cuda
