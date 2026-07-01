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

#include <cstdint>

#include "cuda_ops_api.h"
#include "device_utils.cuh"

// ref to:
// https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu

namespace {

using ::xllm::kernel::cuda::xllm_ldg;

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

// Check if pointer is 16-byte aligned for int4 vectorized access
__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

// Activation and gating kernel template with 128-bit vectorized access
// optimization.
template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void XLLM_KERNEL_ATTR(1024)
    act_and_mul_kernel(scalar_t* __restrict__ out,          // [..., d]
                       const scalar_t* __restrict__ input,  // [..., 2, d]
                       const int d) {
  constexpr int kVecSize = 16 / sizeof(scalar_t);
  const int64_t token_idx = blockIdx.x;
  const scalar_t* x_ptr = input + token_idx * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + token_idx * d;

  // Check alignment for 128-bit vectorized access.
  // All three pointers must be 16-byte aligned for safe int4 operations.
  const bool aligned = is_16byte_aligned(x_ptr) && is_16byte_aligned(y_ptr) &&
                       is_16byte_aligned(out_ptr);

  if (aligned && d >= kVecSize) {
    // Fast path: 128-bit vectorized loop
    const int4* x_vec = reinterpret_cast<const int4*>(x_ptr);
    const int4* y_vec = reinterpret_cast<const int4*>(y_ptr);
    int4* out_vec = reinterpret_cast<int4*>(out_ptr);
    const int num_vecs = d / kVecSize;
    const int vec_end = num_vecs * kVecSize;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 x = xllm_ldg(&x_vec[i]), y = xllm_ldg(&y_vec[i]), r;
      auto* xp = reinterpret_cast<scalar_t*>(&x);
      auto* yp = reinterpret_cast<scalar_t*>(&y);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int j = 0; j < kVecSize; j++) {
        rp[j] = compute<scalar_t, ACT_FN, act_first>(xp[j], yp[j]);
      }
      out_vec[i] = r;
    }
    // Scalar cleanup for remaining elements
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = compute<scalar_t, ACT_FN, act_first>(xllm_ldg(&x_ptr[i]),
                                                        xllm_ldg(&y_ptr[i]));
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = xllm_ldg(&x_ptr[idx]);
      const scalar_t y = xllm_ldg(&y_ptr[idx]);
      out_ptr[idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
    }
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  const float f = static_cast<float>(x);
  return static_cast<T>(f / (1.0f + expf(-f)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = static_cast<float>(x);
  constexpr float kAlpha = M_SQRT1_2;
  return static_cast<T>(f * 0.5f * (1.0f + ::erf(f * kAlpha)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = static_cast<float>(x);
  constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float kKappa = 0.044715;
  float x_cube = f * f * f;
  float inner = kBeta * (f + kKappa * x_cube);
  return static_cast<T>(0.5f * f * (1.0f + ::tanhf(inner)));
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

// Row-major 2D in-place sigmoid-gated multiply.
// `out` and `gate` share logical shape [M, N] with stride(-1) == 1 on both
// (so each row is contiguous in memory) but may have differing row strides.
// One CTA per row, threads stride across the N columns.
template <typename scalar_t>
__global__ void XLLM_KERNEL_ATTR(1024) mul_sigmoid_gate_strided_2d_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ gate,
    const int64_t n,
    const int64_t out_row_stride,
    const int64_t gate_row_stride) {
  const int64_t row = blockIdx.x;
  scalar_t* out_row = out + row * out_row_stride;
  const scalar_t* gate_row = gate + row * gate_row_stride;
  for (int64_t col = threadIdx.x; col < n; col += blockDim.x) {
    // __ldg ok on gate (read-only); out is read+written so use regular load.
    const float g = static_cast<float>(xllm_ldg(&gate_row[col]));
    const float s = 1.0f / (1.0f + expf(-g));
    out_row[col] =
        static_cast<scalar_t>(static_cast<float>(out_row[col]) * s);
  }
}

void launch_mul_sigmoid_gate_inplace(torch::Tensor& out,
                                     const torch::Tensor& gate) {
  const int64_t n = out.numel();
  if (n == 0) {
    return;
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(out));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Collapse to row-major 2D [M, N] where N is the size of the last dim.
  const int64_t last_dim = out.size(-1);
  const int64_t M = n / last_dim;
  const int64_t out_row_stride = (out.dim() <= 1) ? last_dim : out.stride(-2);
  const int64_t gate_row_stride = (gate.dim() <= 1) ? last_dim : gate.stride(-2);

  const int threads = std::min<int64_t>(last_dim, 1024);
  dim3 grid(static_cast<unsigned int>(M));
  dim3 block(static_cast<unsigned int>(threads));
  DISPATCH_FLOATING_TYPES(out.scalar_type(), "mul_sigmoid_gate_inplace", [&] {
    mul_sigmoid_gate_strided_2d_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),
                                     gate.data_ptr<scalar_t>(),
                                     last_dim,
                                     out_row_stride,
                                     gate_row_stride);
  });
}
}  // namespace

namespace xllm::kernel::cuda {

void mul_sigmoid_gate_inplace(torch::Tensor& out, const torch::Tensor& gate) {
  TORCH_CHECK(out.defined() && gate.defined(), "out and gate must be defined");
  TORCH_CHECK(out.sizes() == gate.sizes(), "out and gate must have same shape");
  TORCH_CHECK(out.scalar_type() == gate.scalar_type(), "dtype mismatch");
  TORCH_CHECK(out.device() == gate.device(), "device mismatch");
  TORCH_CHECK(out.dim() >= 1, "out must be at least 1D");
  TORCH_CHECK(out.stride(-1) == 1 && gate.stride(-1) == 1,
              "out and gate must have last-dim stride == 1");
  // For dim > 2 the kernel collapses leading dims into the row index using
  // out.stride(-2)/gate.stride(-2), which is only valid when those leading
  // dims are themselves row-major over the inner [last_dim] block. That is
  // true for the call sites we care about (qkv slices) where everything but
  // the last dim is dense.
  if (out.dim() > 2) {
    const int64_t numel_per_row = out.size(-1);
    TORCH_CHECK(out.numel() % numel_per_row == 0,
                "out shape not collapsible to 2D");
    TORCH_CHECK(gate.numel() % numel_per_row == 0,
                "gate shape not collapsible to 2D");
  }
  launch_mul_sigmoid_gate_inplace(out, gate);
}

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode) {
  if (act_mode != "silu" && act_mode != "gelu" && act_mode != "gelu_tanh" &&
      act_mode != "gelu_pytorch_tanh") {
    LOG(FATAL) << "Unsupported act mode: " << act_mode
               << ", only support silu, gelu, gelu_tanh, gelu_pytorch_tanh";
  }

  // flashinfer act_and_mul ops
  // std::string uri = act_mode + "_and_mul";
  // FunctionFactory::get_instance().act_and_mul(uri).call(
  //     out, input, support_pdl());

  if (act_mode == "silu") {
    silu_and_mul(out, input);
  } else if (act_mode == "gelu") {
    gelu_and_mul(out, input);
  } else if (act_mode == "gelu_tanh" || act_mode == "gelu_pytorch_tanh") {
    // gelu_tanh or gelu_pytorch_tanh (mathematically equivalent)
    gelu_tanh_and_mul(out, input);
  }
}

}  // namespace xllm::kernel::cuda
