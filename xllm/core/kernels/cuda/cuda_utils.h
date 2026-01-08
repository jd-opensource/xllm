/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/cuda_utils.h

#pragma once

#include <stdio.h>

#if defined(__HIPCC__)
#define HOST_DEVICE_INLINE __host__ __device__
#define DEVICE_INLINE __device__
#define HOST_INLINE __host__
#elif defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_INLINE __host__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#define DEVICE_INLINE inline
#define HOST_INLINE inline
#endif

#define CUDA_CHECK(cmd)                         \
  do {                                          \
    cudaError_t e = cmd;                        \
    if (e != cudaSuccess) {                     \
      printf("Failed: Cuda error %s:%d '%s'\n", \
             __FILE__,                          \
             __LINE__,                          \
             cudaGetErrorString(e));            \
      exit(EXIT_FAILURE);                       \
    }                                           \
  } while (0)

namespace xllm {
namespace kernel {
namespace cuda {

int64_t get_device_attribute(int64_t attribute, int64_t device_id);

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace cuda
}  // namespace kernel
}  // namespace xllm