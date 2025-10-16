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

// Mock ProcessGroup for testing
class MockProcessGroup : public xllm::ProcessGroup {
 public:
  MockProcessGroup(int rank, int world_size, const torch::Device& device)
      : xllm::ProcessGroup(device) {}

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
