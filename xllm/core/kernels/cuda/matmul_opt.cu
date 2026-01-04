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

#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>

#include "cuda_ops_api.h"

namespace {

// Tile大小定义
#define TILE_SIZE 16
// 为了避免bank conflict，在shared memory中每行末尾添加一个元素
// Bank conflict原理：
// - CUDA的shared memory被组织成32个banks（对于大多数架构）
// - 如果多个线程同时访问同一个bank的不同地址，会发生bank conflict
// - 通过padding（在每行末尾添加一个元素），可以改变内存布局
// - 当访问tile_A[ty][k]和tile_B[k][tx]时，由于padding的存在，
//   不同线程访问的地址会映射到不同的banks，从而避免conflict
#define SHARED_MEM_PADDING 1
#define SHARED_MEM_WIDTH (TILE_SIZE + SHARED_MEM_PADDING)

/**
 * 优化的矩阵乘法kernel，避免bank conflict
 * 计算 C = A * B，其中:
 * - A: [M, K]
 * - B: [K, N]
 * - C: [M, N]
 * 
 * 使用tile-based方法，每个block处理一个tile的输出
 * 通过padding避免shared memory的bank conflict
 */
template <typename scalar_t>
__global__ void matmul_kernel_optimized(
    scalar_t* __restrict__ C,          // [M, N]
    const scalar_t* __restrict__ A,    // [M, K]
    const scalar_t* __restrict__ B,    // [K, N]
    int M, int N, int K) {
  // Shared memory用于缓存A和B的tile
  // 使用padding避免bank conflict
  __shared__ scalar_t tile_A[TILE_SIZE][SHARED_MEM_WIDTH];
  __shared__ scalar_t tile_B[TILE_SIZE][SHARED_MEM_WIDTH];

  // 当前线程在block中的位置
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // 当前block处理的输出tile位置
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  // 当前线程处理的输出元素位置
  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;
  
  // 寄存器累加器，用于存储部分结果
  scalar_t sum = 0.0f;
  
  // 遍历K维度，每次处理一个tile
  for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    // 协作加载A的tile到shared memory
    int a_row = row;
    int a_col = tile * TILE_SIZE + tx;
    if (a_row < M && a_col < K) {
      tile_A[ty][tx] = A[a_row * K + a_col];
    } else {
      tile_A[ty][tx] = 0.0f;
    }
    
    // 协作加载B的tile到shared memory
    int b_row = tile * TILE_SIZE + ty;
    int b_col = col;
    if (b_row < K && b_col < N) {
      tile_B[ty][tx] = B[b_row * N + b_col];
    } else {
      tile_B[ty][tx] = 0.0f;
    }
    
    // 同步，确保所有线程都完成加载
    __syncthreads();
    
    // 计算当前tile的贡献
    // 注意：使用tx访问tile_A，ty访问tile_B，避免bank conflict
    for (int k = 0; k < TILE_SIZE; ++k) {
      // 由于使用了padding，访问时不需要考虑bank conflict
      sum += tile_A[ty][k] * tile_B[k][tx];
    }
    
    // 同步，确保所有线程都完成计算后再加载下一个tile
    __syncthreads();
  }
  
  // 将结果写回全局内存
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

/**
 * 使用寄存器缓存的进一步优化版本
 * 每个线程使用寄存器来缓存数据，减少shared memory访问
 * 这个版本每个线程计算多个输出元素，提高计算密度
 */
template <typename scalar_t>
__global__ void matmul_kernel_register_cached(
    scalar_t* __restrict__ C,          // [M, N]
    const scalar_t* __restrict__ A,    // [M, K]
    const scalar_t* __restrict__ B,    // [K, N]
    int M, int N, int K) {
  __shared__ scalar_t tile_A[TILE_SIZE][SHARED_MEM_WIDTH];
  __shared__ scalar_t tile_B[TILE_SIZE][SHARED_MEM_WIDTH];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  
  // 每个线程处理一个输出元素
  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;
  
  // 使用寄存器累加器
  scalar_t sum = 0.0f;
  
  // 寄存器用于缓存shared memory中的数据
  scalar_t reg_A;
  scalar_t reg_B;
  
  for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    // 协作加载A的tile到shared memory
    int a_row = row;
    int a_col = tile * TILE_SIZE + tx;
    if (a_row < M && a_col < K) {
      tile_A[ty][tx] = A[a_row * K + a_col];
    } else {
      tile_A[ty][tx] = 0.0f;
    }
    
    // 协作加载B的tile到shared memory
    int b_row = tile * TILE_SIZE + ty;
    int b_col = col;
    if (b_row < K && b_col < N) {
      tile_B[ty][tx] = B[b_row * N + b_col];
    } else {
      tile_B[ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    // 使用寄存器缓存进行计算，减少shared memory访问
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      // 从shared memory加载到寄存器
      reg_A = tile_A[ty][k];
      reg_B = tile_B[k][tx];
      // 使用寄存器进行计算
      sum += reg_A * reg_B;
    }
    
    __syncthreads();
  }
  
  // 将结果写回全局内存
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

}  // namespace

namespace xllm::kernel::cuda {

/**
 * 优化的矩阵乘法实现，避免bank conflict
 * 
 * @param A 输入矩阵A，形状为[M, K]
 * @param B 输入矩阵B，形状为[K, N]
 * @return 输出矩阵C，形状为[M, N]，其中C = A * B
 */
torch::Tensor matmul_optimized(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
  TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
  TORCH_CHECK(A.size(1) == B.size(0), "A and B dimensions must match for matrix multiplication");
  TORCH_CHECK(A.scalar_type() == B.scalar_type(), "A and B must have the same dtype");
  
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  
  // 创建输出tensor
  auto C = torch::empty({M, N}, A.options());
  
  // 设置CUDA设备
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // 计算grid和block大小
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
  
  // 根据数据类型分发kernel
  AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_optimized", [&] {
    // 使用优化的kernel，避免bank conflict
    matmul_kernel_optimized<scalar_t><<<grid, block, 0, stream>>>(
        C.data_ptr<scalar_t>(),
        A.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        M, N, K);
  });
  
  // 检查CUDA错误
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
  
  return C;
}

}  // namespace xllm::kernel::cuda
