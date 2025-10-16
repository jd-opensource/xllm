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

#include <flashinfer/norm.cuh>

#include "cuda_ops_api.h"

using namespace flashinfer;

namespace xllm::kernel::cuda {

void rmsnorm(TensorView output,
             TensorView input,
             TensorView weight,
             double eps,
             bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(1, weight);  // weight: (hidden_size)

  auto input_ndim = input->ndim;
  if (input_ndim == 2) {
    // Normal RMSNorm: [batch_size, hidden_size]
    // Use CTA parallelization for better parallelism
    CHECK_DIM(2, output);
    TVM_FFI_ICHECK_EQ(input->shape[1], weight->shape[0]);
    unsigned int batch_size = input->shape[0];
    unsigned int hidden_size = input->shape[1];
    TVM_FFI_ICHECK_EQ(output->shape[0], batch_size);
    TVM_FFI_ICHECK_EQ(output->shape[1], hidden_size);
    cudaSetDevice(input->device.device_id);
    const cudaStream_t stream = get_stream(input->device);

    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input->dtype, c_type, [&] {
      cudaError_t status = norm::RMSNorm(static_cast<c_type*>(input->data),
                                         static_cast<c_type*>(weight->data),
                                         static_cast<c_type*>(output->data),
                                         batch_size,
                                         hidden_size,
                                         input->strides[0],
                                         output->strides[0],
                                         eps,
                                         enable_pdl,
                                         stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "RMSNorm failed with error code " << cudaGetErrorString(status);
      return true;
    });
  } else if (input_ndim == 3) {
    // QK RMSNorm: [batch_size, num_heads, head_dim]
    // Use warp-level parallization
    CHECK_DIM(3, output);  // output: (batch_size, num_heads, hidden_size)
    TVM_FFI_ICHECK_EQ(input->shape[2], weight->shape[0]);
    unsigned int batch_size = input->shape[0];
    unsigned int num_heads = input->shape[1];
    unsigned int hidden_size = input->shape[2];
    TVM_FFI_ICHECK_EQ(output->shape[0], batch_size);
    TVM_FFI_ICHECK_EQ(output->shape[1], num_heads);
    TVM_FFI_ICHECK_EQ(output->shape[2], hidden_size);

    cudaSetDevice(input->device.device_id);
    const cudaStream_t stream = get_stream(input->device);
    DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input->dtype, c_type, [&] {
      cudaError_t status = norm::QKRMSNorm(static_cast<c_type*>(input->data),
                                           static_cast<c_type*>(weight->data),
                                           static_cast<c_type*>(output->data),
                                           batch_size,
                                           num_heads,
                                           hidden_size,
                                           input->strides[0],
                                           input->strides[1],
                                           output->strides[0],
                                           output->strides[1],
                                           eps,
                                           enable_pdl,
                                           stream);
      TVM_FFI_ICHECK(status == cudaSuccess)
          << "QKRMSNorm failed with error code " << cudaGetErrorString(status);
      return true;
    });
  } else {
    TVM_FFI_ICHECK(false) << "Unsupported input dimension: " << input_ndim;
  }
}

}  // namespace xllm::kernel::cuda