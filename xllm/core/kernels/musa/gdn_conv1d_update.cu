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

// silu(x) = x / (1 + exp(-x)). Computed via __expf for hardware acceleration
// where available. fp32-only.
__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + __expf(-x));
}

// Fused single-token causal-conv1d decode kernel for arbitrary width in
// [2, 5]. Each thread owns one (batch, feat) pair: it reads `width-1` state
// values + the new x_cur, accumulates the conv result, optionally adds
// bias and applies silu, writes the output, and shifts the conv_state ring
// buffer in-place (state[i] <- state[i+1] for i in [0, state_len-1) and
// state[state_len-1] <- x_cur).
//
// Strides are passed in explicitly because the kernel runs on packed
// continuous-batching layouts where the canonical (dim, token) ordering is
// not guaranteed across callers.
template <typename T>
__global__ void __launch_bounds__(256, 1) conv1d_decode_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ conv_state,
    const int32_t* __restrict__ cache_indices,
    T* __restrict__ out,
    int64_t x_stride_token,
    int64_t x_stride_dim,
    int64_t w_stride_dim,
    int64_t w_stride_width,
    int64_t state_stride_seq,
    int64_t state_stride_dim,
    int64_t state_stride_token,
    int64_t out_stride_token,
    int64_t out_stride_dim,
    int batch,
    int dim,
    int num_cache_lines,
    int pad_slot_id,
    int width,
    bool has_bias,
    bool silu_activation) {
  const int batch_idx = blockIdx.y;
  const int feat = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch || feat >= dim) {
    return;
  }
  // cache_indices is required by the host wrapper preconditions.
  const int cache_idx = cache_indices[batch_idx];
  if (cache_idx == pad_slot_id || cache_idx < 0 ||
      cache_idx >= num_cache_lines) {
    return;
  }

  const int64_t x_base = static_cast<int64_t>(batch_idx) * x_stride_token +
                         static_cast<int64_t>(feat) * x_stride_dim;
  const int64_t state_base =
      static_cast<int64_t>(cache_idx) * state_stride_seq +
      static_cast<int64_t>(feat) * state_stride_dim;
  const int64_t w_base = static_cast<int64_t>(feat) * w_stride_dim;
  const int64_t out_base = static_cast<int64_t>(batch_idx) * out_stride_token +
                           static_cast<int64_t>(feat) * out_stride_dim;
  const int state_len = width - 1;

  // Load state into registers. width >= 2 -> state_len >= 1; for Qwen3.5
  // width=4 we use 3 registers. Up to width=5 (state_len=4) handled.
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  if (state_len >= 1) {
    s0 = to_f32<T>(conv_state[state_base + 0 * state_stride_token]);
  }
  if (state_len >= 2) {
    s1 = to_f32<T>(conv_state[state_base + 1 * state_stride_token]);
  }
  if (state_len >= 3) {
    s2 = to_f32<T>(conv_state[state_base + 2 * state_stride_token]);
  }
  if (state_len >= 4) {
    s3 = to_f32<T>(conv_state[state_base + 3 * state_stride_token]);
  }
  const float x_cur = to_f32<T>(x[x_base]);

  // Load weights (width values, taps applied as [state_0, state_1, ...,
  // state_{state_len-1}, x_cur]). For width=4: state taps + x_cur tap.
  float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f, w4 = 0.0f;
  w0 = to_f32<T>(weight[w_base + 0 * w_stride_width]);
  if (width >= 2) {
    w1 = to_f32<T>(weight[w_base + 1 * w_stride_width]);
  }
  if (width >= 3) {
    w2 = to_f32<T>(weight[w_base + 2 * w_stride_width]);
  }
  if (width >= 4) {
    w3 = to_f32<T>(weight[w_base + 3 * w_stride_width]);
  }
  if (width >= 5) {
    w4 = to_f32<T>(weight[w_base + 4 * w_stride_width]);
  }

  // Convolution accumulation: tap-by-tap product with state followed by
  // x_cur for the final tap. Shaped so width=4 hits the fast common case
  // (Qwen3.5 default conv_kernel_size).
  float acc = has_bias ? to_f32<T>(bias[feat]) : 0.0f;
  if (width == 2) {
    acc += s0 * w0 + x_cur * w1;
  } else if (width == 3) {
    acc += s0 * w0 + s1 * w1 + x_cur * w2;
  } else if (width == 4) {
    acc += s0 * w0 + s1 * w1 + s2 * w2 + x_cur * w3;
  } else {
    // width == 5
    acc += s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3 + x_cur * w4;
  }
  if (silu_activation) {
    acc = silu_f32(acc);
  }
  out[out_base] = from_f32<T>(acc);

  // In-place state ring shift: state[i] <- state[i+1], state[last] <- x_cur.
  // Done unconditionally and in fp32-cast-on-store; mirrors sglang's
  // _causal_conv1d_decode_width4_batched_kernel.
  if (state_len >= 1) {
    if (state_len >= 2) {
      conv_state[state_base + 0 * state_stride_token] = from_f32<T>(s1);
    } else {
      conv_state[state_base + 0 * state_stride_token] = from_f32<T>(x_cur);
    }
  }
  if (state_len >= 2) {
    if (state_len >= 3) {
      conv_state[state_base + 1 * state_stride_token] = from_f32<T>(s2);
    } else {
      conv_state[state_base + 1 * state_stride_token] = from_f32<T>(x_cur);
    }
  }
  if (state_len >= 3) {
    if (state_len >= 4) {
      conv_state[state_base + 2 * state_stride_token] = from_f32<T>(s3);
    } else {
      conv_state[state_base + 2 * state_stride_token] = from_f32<T>(x_cur);
    }
  }
  if (state_len >= 4) {
    conv_state[state_base + 3 * state_stride_token] = from_f32<T>(x_cur);
  }
}

template <typename T>
void launch_conv1d_decode(const torch::Tensor& x,
                          const torch::Tensor& weight,
                          const torch::Tensor* bias_or_null,
                          torch::Tensor& conv_state,
                          const torch::Tensor& cache_indices,
                          torch::Tensor& out,
                          int batch,
                          int dim,
                          int num_cache_lines,
                          int pad_slot_id,
                          int width,
                          bool silu_activation,
                          cudaStream_t stream) {
  const int threads = (dim >= 256) ? 256 : ((dim + 31) / 32) * 32;
  const int blocks_x = (dim + threads - 1) / threads;
  dim3 grid(blocks_x, batch);
  dim3 block(threads);
  conv1d_decode_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(weight.data_ptr()),
      bias_or_null ? reinterpret_cast<const T*>(bias_or_null->data_ptr())
                   : nullptr,
      reinterpret_cast<T*>(conv_state.data_ptr()),
      cache_indices.data_ptr<int32_t>(),
      reinterpret_cast<T*>(out.data_ptr()),
      x.stride(0),
      x.stride(1),
      weight.stride(0),
      weight.stride(1),
      conv_state.stride(0),
      conv_state.stride(1),
      conv_state.stride(2),
      out.stride(0),
      out.stride(1),
      batch,
      dim,
      num_cache_lines,
      pad_slot_id,
      width,
      bias_or_null != nullptr,
      silu_activation);
}

}  // namespace

// Host entry point. Decode-only single-token causal conv1d update with the
// following preconditions enforced by the caller (`gdn_ops.cpp`):
//   * x: [num_tokens, dim], num_tokens == batch (one token per sequence).
//   * weight: [dim, width], width in [2, 5].
//   * conv_state: [num_cache_lines, dim, state_len], state_len == width-1.
//   * cache_indices: [batch], int32.
//   * output_buf: [num_tokens, dim], dtype == x.dtype, stride(-1) == 1.
//
// All four tensors must live on the same device. The kernel updates
// conv_state in place (ring shift + new x append).
void causal_conv1d_decode_fused(const torch::Tensor& x,
                                const torch::Tensor& weight,
                                const std::optional<torch::Tensor>& bias,
                                torch::Tensor conv_state,
                                const torch::Tensor& cache_indices,
                                torch::Tensor output_buf,
                                int pad_slot_id,
                                bool silu_activation) {
  TORCH_CHECK(x.dim() == 2, "causal_conv1d_decode_fused: x must be 2D");
  TORCH_CHECK(weight.dim() == 2,
              "causal_conv1d_decode_fused: weight must be 2D [dim, width]");
  TORCH_CHECK(conv_state.dim() == 3,
              "causal_conv1d_decode_fused: conv_state must be 3D");
  TORCH_CHECK(cache_indices.dim() == 1,
              "causal_conv1d_decode_fused: cache_indices must be 1D");
  TORCH_CHECK(cache_indices.scalar_type() == torch::kInt32,
              "causal_conv1d_decode_fused: cache_indices must be int32");
  TORCH_CHECK(output_buf.dim() == 2,
              "causal_conv1d_decode_fused: output_buf must be 2D");
  TORCH_CHECK(x.scalar_type() == output_buf.scalar_type(),
              "causal_conv1d_decode_fused: output_buf dtype must match x");
  TORCH_CHECK(x.scalar_type() == conv_state.scalar_type(),
              "causal_conv1d_decode_fused: conv_state dtype must match x");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "causal_conv1d_decode_fused: weight dtype must match x");

  const int batch = static_cast<int>(x.size(0));
  const int dim = static_cast<int>(x.size(1));
  const int width = static_cast<int>(weight.size(1));
  const int num_cache_lines = static_cast<int>(conv_state.size(0));
  TORCH_CHECK(width >= 2 && width <= 5,
              "causal_conv1d_decode_fused: width must be in [2,5], got ",
              width);
  TORCH_CHECK(weight.size(0) == dim,
              "causal_conv1d_decode_fused: weight dim mismatch");
  TORCH_CHECK(conv_state.size(1) == dim,
              "causal_conv1d_decode_fused: conv_state dim mismatch");
  TORCH_CHECK(conv_state.size(2) == width - 1,
              "causal_conv1d_decode_fused: conv_state.state_len must be "
              "width - 1");
  TORCH_CHECK(cache_indices.size(0) == batch,
              "causal_conv1d_decode_fused: cache_indices length must match "
              "batch (=x.size(0))");
  TORCH_CHECK(output_buf.size(0) == batch && output_buf.size(1) == dim,
              "causal_conv1d_decode_fused: output_buf shape mismatch");
  TORCH_CHECK(output_buf.stride(1) == 1,
              "causal_conv1d_decode_fused: output_buf last dim must be "
              "contiguous (stride==1)");

  const at::cuda::OptionalCUDAGuard guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const torch::Tensor* bias_ptr = nullptr;
  if (bias.has_value() && bias.value().defined()) {
    bias_ptr = &bias.value();
    TORCH_CHECK(bias_ptr->scalar_type() == x.scalar_type(),
                "causal_conv1d_decode_fused: bias dtype must match x");
    TORCH_CHECK(bias_ptr->numel() == dim,
                "causal_conv1d_decode_fused: bias must have shape [dim]");
  }

  switch (x.scalar_type()) {
    case torch::kBFloat16:
      launch_conv1d_decode<__nv_bfloat16>(x,
                                          weight,
                                          bias_ptr,
                                          conv_state,
                                          cache_indices,
                                          output_buf,
                                          batch,
                                          dim,
                                          num_cache_lines,
                                          pad_slot_id,
                                          width,
                                          silu_activation,
                                          stream);
      break;
    case torch::kHalf:
      launch_conv1d_decode<__half>(x,
                                   weight,
                                   bias_ptr,
                                   conv_state,
                                   cache_indices,
                                   output_buf,
                                   batch,
                                   dim,
                                   num_cache_lines,
                                   pad_slot_id,
                                   width,
                                   silu_activation,
                                   stream);
      break;
    case torch::kFloat32:
      launch_conv1d_decode<float>(x,
                                  weight,
                                  bias_ptr,
                                  conv_state,
                                  cache_indices,
                                  output_buf,
                                  batch,
                                  dim,
                                  num_cache_lines,
                                  pad_slot_id,
                                  width,
                                  silu_activation,
                                  stream);
      break;
    default:
      TORCH_CHECK(false,
                  "causal_conv1d_decode_fused: unsupported dtype ",
                  x.scalar_type());
  }
}

}  // namespace xllm::kernel::cuda
