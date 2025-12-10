/* Copyright 2025 The vLLM Authors and The xLLM Authors. All Rights Reserved.

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
#include <torch/cuda.h>

#include <cub/cub.cuh>

#include "cuda_ops_api.h"
// #include "function_factory.h"
#include "type_convert.cuh"

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu

#if CUB_VERSION >= 200800
#include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
using CubMaxOp = cuda::maximum<>;
#else   // if CUB_VERSION < 200800
using CubAddOp = cub::Sum;
using CubMaxOp = cub::Max;
#endif  // CUB_VERSION

namespace {

using namespace xllm::kernel::cuda;

template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * input_stride + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * input_stride + idx];
    out[blockIdx.x * hidden_size + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int64_t vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
    input_v[strided_id] = temp;
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon,
    const int num_tokens,
    const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    input[blockIdx.x * input_stride + idx] =
        ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                \
  DISPATCH_FLOATING_TYPES(                                              \
      input.scalar_type(), "fused_add_rms_norm_kernel", [&] {           \
        fused_add_rms_norm_kernel<scalar_t, width>                      \
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),    \
                                         input_stride,                  \
                                         residual.data_ptr<scalar_t>(), \
                                         weight.data_ptr<scalar_t>(),   \
                                         epsilon,                       \
                                         num_tokens,                    \
                                         hidden_size);                  \
      });

}  // namespace

namespace xllm::kernel::cuda {

// flashinfer rmsnorm ops
// void rmsnorm(torch::Tensor output,
//              torch::Tensor input,
//              torch::Tensor weight,
//              double eps) {
//   FunctionFactory::get_instance().rmsnorm_func("norm").call(
//       output, input, weight, eps, support_pdl());
// }

void rms_norm(torch::Tensor output,  // [..., hidden_size]
              torch::Tensor input,   // [..., hidden_size]
              torch::Tensor weight,  // [hidden_size]
              double eps) {
  CHECK(output.is_contiguous());
  CHECK(input.stride(-1) == 1);
  CHECK(weight.is_contiguous());

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int64_t input_stride = input.stride(-2);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    rms_norm_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(),
                                     input.data_ptr<scalar_t>(),
                                     input_stride,
                                     weight.data_ptr<scalar_t>(),
                                     eps,
                                     num_tokens,
                                     hidden_size);
  });
}

void fused_add_rms_norm(torch::Tensor& input,     // [..., hidden_size]
                        torch::Tensor& residual,  // [..., hidden_size]
                        torch::Tensor& weight,    // [hidden_size]
                        double epsilon) {
  CHECK(weight.scalar_type() == input.scalar_type());
  CHECK(input.scalar_type() == residual.scalar_type());
  CHECK(residual.is_contiguous());
  CHECK(weight.is_contiguous());
  int hidden_size = input.size(-1);
  int64_t input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  constexpr int vector_width = 8;
  constexpr int req_alignment_bytes =
      vector_width * 2;  // vector_width * sizeof(bfloat16 or float16) (float32
                         // falls back to non-vectorized version anyway)
  bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 &&
                          res_ptr % req_alignment_bytes == 0 &&
                          wt_ptr % req_alignment_bytes == 0;
  bool offsets_are_multiple_of_vector_width =
      hidden_size % vector_width == 0 && input_stride % vector_width == 0;
  if (ptrs_are_aligned && offsets_are_multiple_of_vector_width) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}

}  // namespace xllm::kernel::cuda