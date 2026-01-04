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

#pragma once

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"

#define TORCH_VERSION_LESS_THAN(major, minor) \
  (TORCH_VERSION_MAJOR < (major) ||           \
   (TORCH_VERSION_MAJOR == (major) && TORCH_VERSION_MINOR < (minor)))

#if defined(USE_NPU) && TORCH_VERSION_LESS_THAN(2, 7)
#define USE_NPU_HCCL_BACKEND 1
#include <torch_npu/csrc/distributed/ProcessGroupHCCL.hpp>
using MockBackendBase = c10d_npu::ProcessGroupHCCL;
#else
#define USE_NPU_HCCL_BACKEND 0
using MockBackendBase = c10d::Backend;
#endif

namespace xllm {
namespace layer {
namespace test {

namespace detail {

#if USE_NPU_HCCL_BACKEND
inline c10::intrusive_ptr<c10d::TCPStore> createTCPStore(int64_t rank) {
  c10d::TCPStoreOptions opts;
  opts.port = 0;
  opts.isServer = (rank == 0);
  opts.waitWorkers = true;
  return c10::make_intrusive<c10d::TCPStore>("127.0.0.1", opts);
}
#endif

}  // namespace detail

// Mock Backend for testing - minimal implementation for tp=1 tests
class MockBackend : public MockBackendBase {
 public:
#if USE_NPU_HCCL_BACKEND
  MockBackend(int64_t rank, int64_t world_size)
      : MockBackendBase(detail::createTCPStore(rank),
                        rank,
                        world_size,
                        MockBackendBase::Options::create()),
        rank_(rank),
        world_size_(world_size) {}
#else
  MockBackend(int64_t rank, int64_t world_size)
      : MockBackendBase(rank, world_size),
        rank_(rank),
        world_size_(world_size) {}
#endif

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<torch::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override {
    // Mock implementation - return a completed work
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<torch::Tensor>>& outputTensors,
      std::vector<torch::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    // Mock implementation - return a completed work
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<torch::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<torch::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<torch::Tensor>>& outputTensorLists,
      std::vector<torch::Tensor>& inputTensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<torch::Tensor>& outputTensors,
      std::vector<std::vector<torch::Tensor>>& inputTensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      torch::Tensor& outputTensor,
      torch::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<torch::Tensor>& outputTensors,
      std::vector<torch::Tensor>& inputTensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> send(std::vector<torch::Tensor>& tensors,
                                      int dstRank,
                                      int tag) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> recv(std::vector<torch::Tensor>& tensors,
                                      int srcRank,
                                      int tag) override {
    return c10::make_intrusive<c10d::Work>();
  }

  c10::intrusive_ptr<c10d::Work> recvAnysource(
      std::vector<torch::Tensor>& tensors,
      int tag) override {
    return c10::make_intrusive<c10d::Work>();
  }

  int64_t getRank() const { return rank_; }

  int64_t getSize() const { return world_size_; }

#if !TORCH_VERSION_LESS_THAN(2, 7)
  void shutdown() override {
    // Mock implementation - do nothing
  }
#endif

 private:
  int64_t rank_;
  int64_t world_size_;
};

// Mock ProcessGroup for testing
class MockProcessGroup : public xllm::ProcessGroup {
 public:
  MockProcessGroup(const torch::Device& device,
                   int64_t rank = 0,
                   int64_t world_size = 1)
      : xllm::ProcessGroup(device) {
    // Initialize pg_ with a mock backend for testing
    pg_ = std::make_unique<MockBackend>(rank, world_size);
  }

  void allreduce(torch::Tensor& input) override {
    // Mock implementation - do nothing for testing
  }

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) override {
    // Mock implementation - just copy input to outputs
    outputs.resize(this->world_size());
    for (size_t i = 0; i < this->world_size(); ++i) {
      outputs[i] = input.clone();
    }
  }
};

// Helper function to create custom input tensor for precision testing
torch::Tensor create_custom_input(const std::vector<int64_t>& shape,
                                  const std::vector<float>& values,
                                  const torch::TensorOptions& options);

// Helper function to verify tensor values are close to expected
void verify_tensor_close(const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         double rtol = 1e-5,
                         double atol = 1e-8);

// Helper function to verify precision against expected output
void verify_precision(const torch::Tensor& actual_output,
                      const std::vector<float>& expected_values,
                      double rtol = 1e-3,
                      double atol = 1e-4);

// Helper function to create default model arguments for testing
ModelArgs create_default_model_args();

// Helper function to create default quantization arguments for testing
QuantArgs create_default_quant_args();

// Helper function to create default parallel arguments for testing
ParallelArgs create_default_parallel_args(
    std::unique_ptr<xllm::ProcessGroup>& mock_process_group);

// create a tensor with a seeded random number generator (based on key and
// shape) It is robust enough to generate the same tensor across any device or
// os
torch::Tensor seeded_tensor(const std::string& key,
                            torch::IntArrayRef shape,
                            torch::ScalarType dtype = torch::kFloat,
                            torch::Device device = torch::Device(torch::kCPU));

}  // namespace test
}  // namespace layer
}  // namespace xllm
