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

#pragma once

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <mutex>
#include <string>
#include <type_traits>

#ifndef check_cuda_error
#define check_cuda_error(call) C10_CUDA_CHECK(call)
#endif

#if defined(__CUDACC__) || defined(_NVHPC_CUDA)
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_INLINE __host__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#define DEVICE_INLINE inline
#define HOST_INLINE inline
#endif

namespace xllm::kernel::cuda {

inline int getMultiProcessorCount() {
  static int nSM{0};
  static std::once_flag flag;

  std::call_once(flag, []() {
    int deviceID{0};
    check_cuda_error(cudaGetDevice(&deviceID));
    check_cuda_error(
        cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
  });

  return nSM;
}

class NvtxRange {
 public:
  NvtxRange(const std::string& name) { nvtxRangePush(name.c_str()); }

  ~NvtxRange() { nvtxRangePop(); }
};

template <typename T>
HOST_DEVICE_INLINE constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

}  // namespace xllm::kernel::cuda
