/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <flashinfer/activation.cuh>

#include "cuda_ops_api.h"

using namespace flashinfer;

namespace xllm::kernel::cuda {

__device__ __forceinline__ float silu(const float& val) {
  return val / (1.0f + __expf(-val));
}

__device__ __forceinline__ float gelu(const float& val) {
  constexpr float kAlpha = M_SQRT1_2;
  return val * 0.5f * (1.0f + ::erf(val * kAlpha));
}

__device__ __forceinline__ float gelu_tanh(const float& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f *
                                 (val + 0.044715f * val * val * val))));
  return val * cdf;
}

void act_and_mul(TensorView out,
                 TensorView input,
                 const std::string& act_mode,
                 bool enable_pdl) {
  int d = input->shape[input->ndim - 1] / 2;
  int64_t num_tokens = input.numel() / input->shape[input->ndim - 1];
  dim3 grid(num_tokens);

  cudaSetDevice(out->device.device_id);
  const cudaStream_t stream = get_stream(out->device);
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input->dtype, c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    cudaLaunchConfig_t config;
    config.gridDim = num_tokens;
    config.blockDim = std::min(d / vec_size, 1024U);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    auto kernel = activation::act_and_mul_kernel<c_type, act_mode>;

    cudaLaunchKernelEx(&config,
                       kernel,
                       static_cast<c_type*>(out->data),
                       static_cast<c_type*>(input->data),
                       d);

    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess)
        << "Failed to launch kernel: " << cudaGetErrorString(err);

    return true;
  });
}

}  // namespace xllm::kernel::cuda