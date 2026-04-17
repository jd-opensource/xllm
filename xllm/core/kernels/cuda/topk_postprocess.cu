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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <limits>
#include <tuple>

#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/utils.h"

namespace xllm::kernel::cuda {
namespace {

template <typename scalar_t, int32_t kThreads>
__global__ void rec_topk_postprocess_kernel(
    const scalar_t* __restrict__ topk_values,
    const int64_t* __restrict__ topk_indices,
    const float* __restrict__ temperatures,
    bool has_temperatures,
    float* __restrict__ top_logprobs,
    int64_t* __restrict__ next_tokens,
    float* __restrict__ logprobs,
    int32_t topk,
    int64_t stride) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int64_t base = static_cast<int64_t>(row) * stride;

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float row_max;
  __shared__ float log_denom;
  extern __shared__ float scaled_values[];

  float inv_temperature = 1.0f;
  if (has_temperatures) {
    float temperature = temperatures[row];
    if (temperature == 0.0f) {
      temperature = 1.0f;
    }
    inv_temperature = 1.0f / temperature;
  }

  for (int32_t col = static_cast<int32_t>(threadIdx.x); col < topk;
       col += static_cast<int32_t>(blockDim.x)) {
    scaled_values[col] =
        static_cast<float>(topk_values[base + col]) * inv_temperature;
  }
  __syncthreads();

  float thread_max = -std::numeric_limits<float>::infinity();
  for (int32_t col = static_cast<int32_t>(threadIdx.x); col < topk;
       col += static_cast<int32_t>(blockDim.x)) {
    thread_max =
        scaled_values[col] > thread_max ? scaled_values[col] : thread_max;
  }

  float block_max = BlockReduce(reduce_storage).Reduce(thread_max, cub::Max());
  if (threadIdx.x == 0) {
    row_max = block_max;
  }
  __syncthreads();

  float thread_sum = 0.0f;
  for (int32_t col = static_cast<int32_t>(threadIdx.x); col < topk;
       col += static_cast<int32_t>(blockDim.x)) {
    thread_sum += expf(scaled_values[col] - row_max);
  }

  float block_sum = BlockReduce(reduce_storage).Sum(thread_sum);
  if (threadIdx.x == 0) {
    log_denom = logf(block_sum) + row_max;
  }
  __syncthreads();

  for (int32_t col = static_cast<int32_t>(threadIdx.x); col < topk;
       col += static_cast<int32_t>(blockDim.x)) {
    top_logprobs[base + col] = scaled_values[col] - log_denom;
  }

  if (threadIdx.x == 0) {
    next_tokens[row] = topk_indices[base];
    logprobs[row] = top_logprobs[base];
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rec_topk_postprocess(
    const torch::Tensor& topk_values,
    const torch::Tensor& topk_indices,
    const torch::Tensor& temperatures) {
  CHECK(topk_values.is_cuda()) << "rec_topk_postprocess: values must be CUDA";
  CHECK(topk_indices.is_cuda()) << "rec_topk_postprocess: indices must be CUDA";
  CHECK(topk_values.dim() == 2)
      << "rec_topk_postprocess: values must be 2D [B, K]";
  CHECK(topk_indices.dim() == 2)
      << "rec_topk_postprocess: indices must be 2D [B, K]";
  CHECK(topk_values.size(0) == topk_indices.size(0))
      << "rec_topk_postprocess: batch size mismatch";
  CHECK(topk_values.size(1) == topk_indices.size(1))
      << "rec_topk_postprocess: topk size mismatch";
  CHECK(topk_indices.scalar_type() == torch::kLong)
      << "rec_topk_postprocess: indices must be int64";

  const int64_t batch64 = topk_values.size(0);
  const int64_t topk64 = topk_values.size(1);
  CHECK(batch64 >= 0 && batch64 <= INT32_MAX)
      << "rec_topk_postprocess: batch too large";
  CHECK(topk64 > 0 && topk64 <= INT32_MAX)
      << "rec_topk_postprocess: topk too large";

  const int32_t batch_size = static_cast<int32_t>(batch64);
  const int32_t topk = static_cast<int32_t>(topk64);

  if (batch_size == 0) {
    torch::Tensor empty_next_tokens =
        torch::empty({batch64},
                     torch::TensorOptions()
                         .dtype(torch::kInt64)
                         .device(topk_values.device()));
    torch::Tensor empty_logprobs =
        torch::empty({batch64},
                     torch::TensorOptions()
                         .dtype(torch::kFloat32)
                         .device(topk_values.device()));
    torch::Tensor empty_top_logprobs =
        torch::empty({batch64, topk64},
                     torch::TensorOptions()
                         .dtype(torch::kFloat32)
                         .device(topk_values.device()));
    return std::make_tuple(
        empty_next_tokens, empty_logprobs, empty_top_logprobs);
  }

  bool has_temperatures = temperatures.defined();
  torch::Tensor contiguous_temperatures = temperatures;
  if (has_temperatures) {
    CHECK(contiguous_temperatures.is_cuda())
        << "rec_topk_postprocess: temperatures must be CUDA";
    CHECK(contiguous_temperatures.dim() == 1)
        << "rec_topk_postprocess: temperatures must be 1D [B]";
    CHECK(contiguous_temperatures.size(0) == batch64)
        << "rec_topk_postprocess: temperatures size mismatch";
    CHECK(contiguous_temperatures.scalar_type() == torch::kFloat32)
        << "rec_topk_postprocess: temperatures must be float32";
    contiguous_temperatures = contiguous_temperatures.contiguous();
  }

  c10::cuda::CUDAGuard device_guard(topk_values.device());
  torch::Tensor contiguous_values = topk_values.contiguous();
  torch::Tensor contiguous_indices = topk_indices.contiguous();
  torch::Tensor output_top_logprobs =
      torch::empty({batch64, topk64},
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(topk_values.device()));
  torch::Tensor output_next_tokens = torch::empty(
      {batch64},
      torch::TensorOptions().dtype(torch::kInt64).device(topk_values.device()));
  torch::Tensor output_logprobs =
      torch::empty({batch64},
                   torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(topk_values.device()));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t stride = topk64;
  constexpr int32_t kThreads = 256;
  const dim3 grid(batch_size);
  const dim3 block(std::min<int32_t>(kThreads, 1024));
  const size_t dynamic_shared_memory_bytes =
      static_cast<size_t>(topk) * sizeof(float);

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  constexpr size_t static_shared_memory_bytes =
      sizeof(typename BlockReduce::TempStorage) + 2 * sizeof(float);
  const size_t total_shared_memory_bytes =
      dynamic_shared_memory_bytes + static_shared_memory_bytes;

  int32_t max_shared_memory_per_block = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_memory_per_block,
                                        cudaDevAttrMaxSharedMemoryPerBlock,
                                        topk_values.get_device()));
  CHECK(total_shared_memory_bytes <=
        static_cast<size_t>(max_shared_memory_per_block))
      << "rec_topk_postprocess: topk (" << topk << ") requires "
      << total_shared_memory_bytes << " bytes shared memory, but device only "
      << "supports " << max_shared_memory_per_block;

  DISPATCH_FLOATING_TYPES(
      contiguous_values.scalar_type(), "rec_topk_postprocess", [&] {
        rec_topk_postprocess_kernel<scalar_t, kThreads>
            <<<grid, block, dynamic_shared_memory_bytes, stream>>>(
                contiguous_values.data_ptr<scalar_t>(),
                contiguous_indices.data_ptr<int64_t>(),
                has_temperatures ? contiguous_temperatures.data_ptr<float>()
                                 : nullptr,
                has_temperatures,
                output_top_logprobs.data_ptr<float>(),
                output_next_tokens.data_ptr<int64_t>(),
                output_logprobs.data_ptr<float>(),
                topk,
                stride);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(
      output_next_tokens, output_logprobs, output_top_logprobs);
}

}  // namespace xllm::kernel::cuda
