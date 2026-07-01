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

template <typename T>
__device__ __forceinline__ float to_f32(T v) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(v);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(v);
  } else {
    return static_cast<float>(v);
  }
}

template <typename T>
__device__ __forceinline__ T from_f32(float v) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(v);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16_rn(v);
  } else {
    return static_cast<T>(v);
  }
}

// fast sigmoid(z) = 1 / (1 + exp2(-z * log2(e))) matches sglang's
// rms_norm_gated_kernel exactly.
__device__ __forceinline__ float fast_sigmoid_f32(float z) {
  constexpr float kLog2e = 1.4426950408889634f;
  return 1.0f / (1.0f + exp2f(-z * kLog2e));
}

// Fused gated RMSNorm. Grid: (M,). Block: (threads,) where threads is a
// power-of-two between 32 and 1024, chosen by the launcher to be >= N when
// possible. Each block handles one row of [M, N] and uses shared-memory
// reduction for the sum-of-squares pass.
template <typename T, int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS, 1)
    gated_rms_norm_kernel(const T* __restrict__ x,
                          const T* __restrict__ w,
                          const T* __restrict__ z,
                          T* __restrict__ y,
                          int64_t x_stride_row,
                          int64_t z_stride_row,
                          int64_t y_stride_row,
                          int N,
                          float eps,
                          float inv_N) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const T* __restrict__ x_row = x + static_cast<int64_t>(row) * x_stride_row;
  const T* __restrict__ z_row = z + static_cast<int64_t>(row) * z_stride_row;
  T* __restrict__ y_row = y + static_cast<int64_t>(row) * y_stride_row;

  // Pass 1: sum of squares with strided coverage of the row.
  float local_sum = 0.0f;
  for (int n = tid; n < N; n += BLOCK_THREADS) {
    const float x_val = to_f32<T>(x_row[n]);
    local_sum += x_val * x_val;
  }

  // Cross-thread reduction via shared memory (portable across MUSA SIMD
  // widths; matches sglang_qwen35/.../rmsnorm.mu's `block_sum` style rather
  // than warp shuffles).
  __shared__ float reduce[BLOCK_THREADS];
  reduce[tid] = local_sum;
  __syncthreads();
#pragma unroll
  for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] += reduce[tid + stride];
    }
    __syncthreads();
  }

  // Broadcast inv_rms to all threads via shared memory.
  __shared__ float inv_rms_shared;
  if (tid == 0) {
    inv_rms_shared = rsqrtf(reduce[0] * inv_N + eps);
  }
  __syncthreads();
  const float inv_rms = inv_rms_shared;

  // Pass 2: per-element compute. y = (x * inv_rms * w) * (z * sigmoid(z)).
  for (int n = tid; n < N; n += BLOCK_THREADS) {
    const float x_val = to_f32<T>(x_row[n]);
    const float w_val = to_f32<T>(w[n]);
    const float z_val = to_f32<T>(z_row[n]);
    const float normed = x_val * inv_rms * w_val;
    const float gate = z_val * fast_sigmoid_f32(z_val);
    y_row[n] = from_f32<T>(normed * gate);
  }
}

// Pick a block size that's a power of two, >= 32, and ideally >= N (so each
// thread handles a single element). Falls back to 1024 for very large N.
inline int pick_block_threads(int N) {
  // Round up to next power of two, clamped to [32, 1024].
  int bt = 32;
  while (bt < N && bt < 1024) {
    bt <<= 1;
  }
  return bt;
}

#define GATED_RMSNORM_DISPATCH_BLOCK(T_TYPE, BT)                              \
  do {                                                                       \
    gated_rms_norm_kernel<T_TYPE, BT><<<rows, BT, 0, stream>>>(               \
        reinterpret_cast<const T_TYPE*>(x.data_ptr()),                       \
        reinterpret_cast<const T_TYPE*>(weight.data_ptr()),                  \
        reinterpret_cast<const T_TYPE*>(z.data_ptr()),                       \
        reinterpret_cast<T_TYPE*>(output.data_ptr()),                        \
        x.stride(0),                                                         \
        z.stride(0),                                                         \
        output.stride(0),                                                    \
        N,                                                                   \
        static_cast<float>(eps),                                             \
        inv_N);                                                              \
  } while (0)

template <typename T>
void launch_gated_rms_norm(const torch::Tensor& x,
                           const torch::Tensor& weight,
                           const torch::Tensor& z,
                           torch::Tensor& output,
                           int N,
                           double eps,
                           cudaStream_t stream) {
  const int rows = static_cast<int>(x.size(0));
  const int block_threads = pick_block_threads(N);
  const float inv_N = 1.0f / static_cast<float>(N);
  switch (block_threads) {
    case 32:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 32);
      break;
    case 64:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 64);
      break;
    case 128:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 128);
      break;
    case 256:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 256);
      break;
    case 512:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 512);
      break;
    default:
      GATED_RMSNORM_DISPATCH_BLOCK(T, 1024);
      break;
  }
}

#undef GATED_RMSNORM_DISPATCH_BLOCK

}  // namespace

// Public host entry point. The caller (RmsNormGatedImpl::forward via
// cuda::gated_layer_norm in gdn_ops.cpp) must ensure:
//   * x, z, output: [M, N] (reshape to 2D ahead of time), same dtype, same
//     device. x.stride(-1) == z.stride(-1) == output.stride(-1) == 1.
//   * weight: [N], same dtype as x, contiguous.
//   * dtype: fp32 / fp16 / bf16.
//   * The kernel writes directly into `output` (no allocations).
void gated_rms_norm_fused(const torch::Tensor& x,
                          const torch::Tensor& weight,
                          const torch::Tensor& z,
                          torch::Tensor output,
                          double eps) {
  TORCH_CHECK(x.dim() == 2, "gated_rms_norm_fused: x must be 2D [M, N]");
  TORCH_CHECK(z.dim() == 2, "gated_rms_norm_fused: z must be 2D [M, N]");
  TORCH_CHECK(output.dim() == 2,
              "gated_rms_norm_fused: output must be 2D [M, N]");
  TORCH_CHECK(weight.dim() == 1, "gated_rms_norm_fused: weight must be 1D [N]");
  TORCH_CHECK(x.size(0) == z.size(0) && x.size(0) == output.size(0),
              "gated_rms_norm_fused: row count mismatch");
  TORCH_CHECK(x.size(1) == z.size(1) && x.size(1) == output.size(1),
              "gated_rms_norm_fused: column count mismatch");
  TORCH_CHECK(weight.size(0) == x.size(1),
              "gated_rms_norm_fused: weight size mismatch");
  TORCH_CHECK(x.scalar_type() == z.scalar_type() &&
                  x.scalar_type() == output.scalar_type() &&
                  x.scalar_type() == weight.scalar_type(),
              "gated_rms_norm_fused: dtype mismatch");
  TORCH_CHECK(x.stride(-1) == 1 && z.stride(-1) == 1 &&
                  output.stride(-1) == 1 && weight.stride(0) == 1,
              "gated_rms_norm_fused: last dim must be contiguous (stride==1)");

  const int N = static_cast<int>(x.size(1));
  TORCH_CHECK(N > 0,
              "gated_rms_norm_fused: hidden_size must be > 0 (got ", N, ")");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (x.scalar_type()) {
    case torch::kBFloat16:
      launch_gated_rms_norm<__nv_bfloat16>(x, weight, z, output, N, eps,
                                            stream);
      break;
    case torch::kHalf:
      launch_gated_rms_norm<__half>(x, weight, z, output, N, eps, stream);
      break;
    case torch::kFloat32:
      launch_gated_rms_norm<float>(x, weight, z, output, N, eps, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "gated_rms_norm_fused: unsupported dtype ", x.scalar_type());
  }
}

}  // namespace xllm::kernel::cuda
