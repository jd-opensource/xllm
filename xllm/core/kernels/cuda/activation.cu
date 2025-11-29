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

#include "cuda_ops_api.h"
#include "function_factory.h"

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu

namespace {

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = *(&input[token_idx * 2 * d + idx]);
    const scalar_t y = *(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                   \
  int d = input.size(-1) / 2;                                              \
  int64_t num_tokens = input.numel() / input.size(-1);                     \
  dim3 grid(num_tokens);                                                   \
  dim3 block(std::min(d, 1024));                                           \
  if (num_tokens == 0) {                                                   \
    return;                                                                \
  }                                                                        \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));        \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();            \
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "act_and_mul_kernel", [&] { \
    act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>              \
        <<<grid, block, 0, stream>>>(                                      \
            out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);      \
  });

void silu_and_mul(torch::Tensor out,    // [..., d]
                  torch::Tensor input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(silu_kernel, true);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(gelu_kernel, true);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(gelu_tanh_kernel, true);
}
}  // namespace

namespace xllm::kernel::cuda {

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode) {
  if (act_mode != "silu" && act_mode != "gelu" && act_mode != "gelu_tanh") {
    LOG(FATAL) << "Unsupported act mode: " << act_mode
               << ", only support silu, gelu, gelu_tanh";
  }

  // flashinfer act_and_mul ops
  // std::string uri = act_mode + "_and_mul";
  // FunctionFactory::get_instance().act_and_mul(uri).call(
  //     out, input, support_pdl());

  if (act_mode == "silu") {
    silu_and_mul(out, input);
  } else if (act_mode == "gelu") {
    gelu_and_mul(out, input);
  } else if (act_mode == "gelu_tanh") {
    gelu_tanh_and_mul(out, input);
  }
}

}  // namespace xllm::kernel::cuda
