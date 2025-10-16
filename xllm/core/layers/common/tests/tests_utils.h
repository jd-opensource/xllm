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

#include <unordered_map>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {
namespace test {

// Mock Backend for testing - minimal implementation for tp=1 tests
class MockBackend : public c10d::Backend {
 public:
  MockBackend(int rank, int world_size)
      : c10d::Backend(rank, world_size), rank_(rank), world_size_(world_size) {}

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

  int getRank() const { return rank_; }

  int getSize() const { return world_size_; }

#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 7
  void shutdown() override {
    // Mock implementation - do nothing
  }
#endif

 private:
  int rank_;
  int world_size_;
};

// Mock ProcessGroup for testing
class MockProcessGroup : public xllm::ProcessGroup {
 public:
  MockProcessGroup(const torch::Device& device,
                   int rank = 0,
                   int world_size = 1)
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
    for (int i = 0; i < this->world_size(); ++i) {
      outputs[i] = input.clone();
    }
  }
};

// Helper function to create all-ones tensor
torch::Tensor CreateOnesTensor(const std::vector<int64_t>& shape,
                               const torch::TensorOptions& options);

// Helper function to create full tensor with specific value
torch::Tensor CreateFullTensor(const std::vector<int64_t>& shape,
                               float value,
                               const torch::TensorOptions& options);

// Helper function to create custom input tensor for precision testing
torch::Tensor CreateCustomInput(const std::vector<int64_t>& shape,
                                const std::vector<float>& values,
                                const torch::TensorOptions& options);

// Helper function to create custom residual tensor for precision testing
torch::Tensor CreateCustomResidual(const std::vector<int64_t>& shape,
                                   const std::vector<float>& values,
                                   const torch::TensorOptions& options);

// Helper function to verify tensor values are close to expected
void VerifyTensorClose(const torch::Tensor& actual,
                       const torch::Tensor& expected,
                       double rtol = 1e-5,
                       double atol = 1e-8);

// Helper function to verify precision against expected output
void VerifyPrecision(const torch::Tensor& actual_output,
                     const std::vector<float>& expected_values,
                     double rtol = 1e-3,
                     double atol = 1e-4);

// Helper function to create default model arguments for testing
ModelArgs CreateDefaultModelArgs();

// Helper function to create default quantization arguments for testing
QuantArgs CreateDefaultQuantArgs();

// Helper function to create default tensor options for testing
torch::TensorOptions CreateDefaultTensorOptions();

// Helper function to create default parallel arguments for testing
ParallelArgs CreateDefaultParallelArgs(
    std::unique_ptr<xllm::ProcessGroup>& mock_process_group);

}  // namespace test
}  // namespace layer
}  // namespace xllm
