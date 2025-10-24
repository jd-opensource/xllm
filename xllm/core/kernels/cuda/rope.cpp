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

#include <flashinfer/pos_enc.cuh>

#include "cuda_ops_api.h"

using namespace flashinfer;
using tvm::ffi::Tensor;

namespace xllm::kernel::cuda {

void apply_rope_pos_ids_cos_sin_cache(TensorView q,
                                      TensorView k,
                                      TensorView q_rope,
                                      TensorView k_rope,
                                      TensorView cos_sin_cache,
                                      TensorView pos_ids,
                                      bool interleave) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k);
  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  CHECK_DEVICE(q, k);
  CHECK_DEVICE(q, cos_sin_cache);
  CHECK_DEVICE(q, pos_ids);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)
  // cos_sin_cache: (max_seq_len, R)
  // First half of R is cos, second half is sin
  CHECK_DIM(2, cos_sin_cache);
  TVM_FFI_ICHECK_EQ(q->shape[0], k->shape[0]);
  TVM_FFI_ICHECK_EQ(q->shape[2], k->shape[2]);
  unsigned int rotary_dim = cos_sin_cache->shape[1];
  unsigned int num_qo_heads = q->shape[1];
  unsigned int num_kv_heads = k->shape[1];
  unsigned int head_dim = q->shape[2];
  unsigned int nnz = q->shape[0];
  size_t q_stride_n = q->strides[0];
  size_t q_stride_h = q->strides[1];
  size_t k_stride_n = k->strides[0];
  size_t k_stride_h = k->strides[1];
  size_t q_rope_stride_n = q_rope->strides[0];
  size_t q_rope_stride_h = q_rope->strides[1];
  size_t k_rope_stride_n = k_rope->strides[0];
  size_t k_rope_stride_h = k_rope->strides[1];

  cudaSetDevice(q->device.device_id);
  const cudaStream_t stream = get_stream(q->device);
  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(q->dtype, c_type, [&] {
    return DISPATCH_DLPACK_IDTYPE_TO_CTYPE(pos_ids->dtype, c_idtype, [&] {
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache(
          static_cast<c_type*>(q->data),
          static_cast<c_type*>(k->data),
          static_cast<c_type*>(q_rope->data),
          static_cast<c_type*>(k_rope->data),
          static_cast<float*>(cos_sin_cache->data),
          static_cast<c_idtype*>(pos_ids->data),
          nnz,
          num_qo_heads,
          num_kv_heads,
          rotary_dim,
          head_dim,
          q_stride_n,
          q_stride_h,
          k_stride_n,
          k_stride_h,
          q_rope_stride_n,
          q_rope_stride_h,
          k_rope_stride_n,
          k_rope_stride_h,
          interleave,
          stream);

      TVM_FFI_ICHECK(status == cudaSuccess)
          << "BatchQKApplyRotaryPosIdsCosSinCache failed with error code "
          << cudaGetErrorString(status);
      return true;
    });
  });
}

}  // namespace xllm::kernel::cuda