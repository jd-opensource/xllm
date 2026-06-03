/* Copyright 2025 The vLLM Authors and The xLLM Authors.
 * SPDX-License-Identifier: Apache-2.0 ILU dtype ↔ CUDA intrinsics. Minimal
 * includes (no torch/all.h — avoids glog macro clashes with c10).
 *
 * BFloat16 must stay defined on every NVCC/ILU compilation pass (host +
 * device): gating the whole struct on __CUDA_ARCH__ >= 800 breaks ILU targets
 * whose
 * __CUDA_ARCH__ is not NVIDIA SM numbering (e.g. ivcore11) while dispatch still
 * instantiates kernels with BFloat16 cache dtypes. Kernels may still
 * early-return BF16 paths on unsupported hardware via fused_qknorm_rope.cu
 * guards.
 */
#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace xllm::kernel::ilu {

template <typename torch_type>
struct _typeConvert {
  static constexpr bool exists = false;
};

template <>
struct _typeConvert<float> {
  static constexpr bool exists = true;
  using hip_type = float;
  using packed_hip_type = float2;
  using packed_hip_type4 = float4;

  __device__ static __forceinline__ float convert(hip_type x) { return x; }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return x;
  }
  __device__ static __forceinline__ float4 convert(packed_hip_type4 x) {
    return x;
  }
};

template <>
struct _typeConvert<c10::Half> {
  static constexpr bool exists = true;
  using hip_type = __half;
  using packed_hip_type = __half2;

  __device__ static __forceinline__ float convert(hip_type x) {
    return __half2float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return __half22float2(x);
  }
  __device__ static __forceinline__ hip_type convert(float x) {
    return __float2half_rn(x);
  }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) {
    return __float22half2_rn(x);
  }
};

template <>
struct _typeConvert<c10::BFloat16> {
  static constexpr bool exists = true;
  using hip_type = __nv_bfloat16;
  using packed_hip_type = __nv_bfloat162;

  __device__ static __forceinline__ float convert(hip_type x) {
    return __bfloat162float(x);
  }
  __device__ static __forceinline__ float2 convert(packed_hip_type x) {
    return __bfloat1622float2(x);
  }
  __device__ static __forceinline__ hip_type convert(float x) {
    return __float2bfloat16(x);
  }
  __device__ static __forceinline__ packed_hip_type convert(float2 x) {
    return __float22bfloat162_rn(x);
  }
};

}  // namespace xllm::kernel::ilu
