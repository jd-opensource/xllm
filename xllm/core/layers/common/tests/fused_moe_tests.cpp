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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/fused_moe.h"
#include "platform/device.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {

class FusedMoETest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize default model arguments for testing
    model_args_ = test::CreateDefaultModelArgs();

    // Initialize w8a8 quantization arguments
    quant_args_ = test::CreateDefaultQuantArgs();

    // Initialize tensor options
    options_ = test::CreateDefaultTensorOptions();

    // Create mock ProcessGroup and initialize ParallelArgs
    parallel_args_ = test::CreateDefaultParallelArgs(mock_process_group_);

    // Note: FusedMoE will be created by individual test cases with their
    // desired dimensions
  }

  void TearDown() override {
    // Clean up if needed
  }

  // Helper function to create all-ones tensor
  torch::Tensor CreateOnesTensor(const std::vector<int64_t>& shape) {
    return test::CreateOnesTensor(shape, options_);
  }

  // Helper function to create all-ones tensor with specific values
  torch::Tensor CreateFullTensor(const std::vector<int64_t>& shape,
                                 float value) {
    return test::CreateFullTensor(shape, value, options_);
  }

  // Helper function to create router logits tensor
  torch::Tensor CreateRouterLogits(const std::vector<int64_t>& shape,
                                   const std::vector<float>& values) {
    return test::CreateCustomInput(shape, values, options_);
  }

  std::unordered_map<std::string, torch::Tensor> CreateDefaultTestWeights(
      int64_t num_experts,
      int64_t hidden_size,
      int64_t intermediate_size) {
    // Create test weights for each expert
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
      std::string expert_prefix = "experts." + std::to_string(expert_id) + ".";

      // Create gate_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto gate_weight =
          CreateFullTensor({intermediate_size, hidden_size}, 2.0f);
      auto gate_qweight = gate_weight.to(torch::kInt8);
      auto gate_scale = torch::full({intermediate_size},
                                    0.1f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
      auto gate_smooth = torch::full({hidden_size},
                                     0.05f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

      // Create up_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto up_weight = CreateFullTensor({intermediate_size, hidden_size}, 2.0f);
      auto up_qweight = up_weight.to(torch::kInt8);
      auto up_scale = torch::full({intermediate_size},
                                  0.1f,
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(options_.device()));
      auto up_smooth = torch::full({hidden_size},
                                   0.05f,
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(options_.device()));

      // Create down_proj weights (RowParallelLinear)
      // Shape: [hidden_size, intermediate_size]
      auto down_weight =
          CreateFullTensor({hidden_size, intermediate_size}, 3.0f);
      auto down_qweight = down_weight.to(torch::kInt8);
      auto down_scale = torch::full({hidden_size},
                                    0.1f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
      auto down_smooth = torch::full({intermediate_size},
                                     0.05f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));
      // Add weights to dictionary
      // expert
      weight_dict[expert_prefix + "gate_proj.qweight"] = gate_qweight;
      weight_dict[expert_prefix + "gate_proj.per_channel_scale"] = gate_scale;
      weight_dict[expert_prefix + "gate_proj.smooth"] = gate_smooth;

      weight_dict[expert_prefix + "up_proj.qweight"] = up_qweight;
      weight_dict[expert_prefix + "up_proj.per_channel_scale"] = up_scale;
      weight_dict[expert_prefix + "up_proj.smooth"] = up_smooth;

      weight_dict[expert_prefix + "down_proj.qweight"] = down_qweight;
      weight_dict[expert_prefix + "down_proj.per_channel_scale"] = down_scale;
      weight_dict[expert_prefix + "down_proj.smooth"] = down_smooth;
    }

    // gate weight generation
    auto gate_weight = CreateFullTensor({num_experts, hidden_size}, 5.0f);
    auto e_score_correction_bias = CreateFullTensor({num_experts}, 0.1f);

    // gate
    weight_dict["gate.weight"] = gate_weight;
    weight_dict["gate.e_score_correction_bias"] = e_score_correction_bias;

    LOG(INFO) << "Test w8a8 smoothquant weights created successfully for "
              << num_experts << " experts";
    LOG(INFO) << "Hidden size: " << hidden_size
              << ", Intermediate size: " << intermediate_size;

    return weight_dict;
  }

  // Helper function to create FusedMoE with custom dimensions
  FusedMoE CreateFusedMoE(int num_experts,
                          int top_k,
                          int num_expert_group,
                          int topk_group,
                          double route_scale,
                          int64_t hidden_size,
                          int64_t intermediate_size,
                          int n_shared_experts = 1,
                          bool is_gated = true,
                          bool has_score_bias = false,
                          bool has_bias = false,
                          bool skip_bias_add = false,
                          int renormalize = 0,
                          const std::string& hidden_act = "silu",
                          const std::string& scoring_func = "sigmoid",
                          const std::string& topk_method = "noaux_tc") {
    return FusedMoE(FusedMoEImpl(num_experts,
                                 top_k,
                                 num_expert_group,
                                 topk_group,
                                 route_scale,
                                 hidden_size,
                                 intermediate_size,
                                 n_shared_experts,
                                 is_gated,
                                 has_score_bias,
                                 has_bias,
                                 skip_bias_add,
                                 renormalize,
                                 hidden_act,
                                 scoring_func,
                                 topk_method,
                                 quant_args_,
                                 parallel_args_,
                                 options_));
  }

  // Helper function to create test weights for the FusedMoE (w8a8 smoothquant
  // format)
  std::unordered_map<std::string, torch::Tensor> CreateTestWeights(
      int num_experts,
      int64_t custom_hidden_size = -1,
      int64_t custom_intermediate_size = -1) {
    // Use custom sizes if provided, otherwise use model_args_ values
    int64_t test_hidden_size = (custom_hidden_size > 0)
                                   ? custom_hidden_size
                                   : model_args_.hidden_size();
    int64_t test_intermediate_size = (custom_intermediate_size > 0)
                                         ? custom_intermediate_size
                                         : model_args_.intermediate_size();

    return CreateDefaultTestWeights(
        num_experts, test_hidden_size, test_intermediate_size);
  }

  // Helper function to verify tensor values are close to expected
  void VerifyTensorClose(const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         double rtol = 1e-5,
                         double atol = 1e-8) {
    test::VerifyTensorClose(actual, expected, rtol, atol);
  }

  // Helper function to create custom input tensor for precision testing
  torch::Tensor CreateCustomInput(const std::vector<int64_t>& shape,
                                  const std::vector<float>& values) {
    return test::CreateCustomInput(shape, values, options_);
  }

  // Helper function to create custom residual tensor for precision testing
  torch::Tensor CreateCustomResidual(const std::vector<int64_t>& shape,
                                     const std::vector<float>& values) {
    return test::CreateCustomResidual(shape, values, options_);
  }

  // Helper function to set expected output for precision verification
  void SetExpectedOutput(const std::vector<float>& expected_values) {
    expected_output_ = expected_values;
  }

  // Helper function to verify precision against expected output
  void VerifyPrecision(const torch::Tensor& actual_output,
                       double rtol = 1e-3,
                       double atol = 1e-4) {
    test::VerifyPrecision(actual_output, expected_output_, rtol, atol);
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;

  // Helper to create a mock ProcessGroup for testing
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;

  // Expected output for precision verification
  std::vector<float> expected_output_;
};

TEST_F(FusedMoETest, LoadStateDictTest) {
  // Test loading weights into the FusedMoE
  const int64_t batch_size = 16;
  const int64_t seq_len = 32;
  const int64_t hidden_size = 7168;
  const int64_t intermediate_size = 2048;
  const int num_experts = 16;
  const int num_expert_group = 4;
  const int topk_group = 4;
  const int top_k = 2;
  const double route_scale = 2.5;
  const bool gated = true;
  const bool has_score_bias = true;
  const bool has_bias = false;
  const bool skip_bias_add = false;
  const int renormalize = 1;
  const int n_shared_experts = 1;

  // Create FusedMoE with default dimensions
  auto fused_moe = CreateFusedMoE(num_experts,
                                  top_k,
                                  num_expert_group,
                                  topk_group,
                                  route_scale,
                                  hidden_size,
                                  intermediate_size,
                                  n_shared_experts,
                                  gated,
                                  has_score_bias,
                                  has_bias,
                                  skip_bias_add,
                                  renormalize);

  // Create test weights and load them
  auto weight_dict =
      CreateTestWeights(num_experts, hidden_size, intermediate_size);

  // Load weights into the FusedMoE
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  // Create input tensors
  auto hidden_states = CreateCustomInput(
      {batch_size, seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  // Create router logits (batch_size * seq_len, num_experts)
  std::vector<float> router_values;
  router_values.reserve(batch_size * seq_len * num_experts);
  for (int i = 0; i < batch_size * seq_len; ++i) {
    for (int j = 0; j < num_experts; ++j) {
      router_values.push_back(static_cast<float>(j) * 0.1f);
    }
  }
  auto router_logits =
      CreateRouterLogits({batch_size * seq_len, num_experts}, router_values);
  auto score_bias = CreateFullTensor({num_experts}, 0.1f);
  auto output =
      fused_moe->forward_expert(hidden_states, router_logits, std::nullopt);

  // Verify output shape
  ASSERT_EQ(output.sizes().size(), 3) << "Output should be 3D tensor";
  ASSERT_EQ(output.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.size(1), seq_len) << "Sequence length should match";
  ASSERT_EQ(output.size(2), hidden_size) << "Hidden size should match";

  // Verify output is not all zeros (weights were loaded)
  auto output_sum = torch::sum(output).item<float>();
  ASSERT_NE(output_sum, 0.0f)
      << "Output should not be all zeros after loading weights";

  LOG(INFO) << "State dict loading test passed - output sum: " << output_sum;
}

TEST_F(FusedMoETest, PrecisionVerificationTest) {
  // Test loading weights into the FusedMoE
  const int64_t batch_size = 16;
  const int64_t seq_len = 32;
  const int64_t hidden_size = 7168;
  const int64_t intermediate_size = 2048;
  const int num_experts = 16;
  const int num_expert_group = 4;
  const int topk_group = 4;
  const int top_k = 2;
  const double route_scale = 2.5;
  const bool gated = true;
  const bool has_score_bias = true;
  const bool has_bias = false;
  const bool skip_bias_add = false;
  const int renormalize = 1;
  const int n_shared_experts = 1;

  // Create FusedMoE with default dimensions
  auto fused_moe = CreateFusedMoE(num_experts,
                                  top_k,
                                  num_expert_group,
                                  topk_group,
                                  route_scale,
                                  hidden_size,
                                  intermediate_size,
                                  n_shared_experts,
                                  gated,
                                  has_score_bias,
                                  has_bias,
                                  skip_bias_add,
                                  renormalize);

  // Create test weights and load them
  auto weight_dict =
      CreateTestWeights(num_experts, hidden_size, intermediate_size);

  // Load weights into the FusedMoE
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  // Create input tensors
  auto hidden_states = CreateCustomInput(
      {batch_size, seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  // Create router logits (batch_size * seq_len, num_experts)
  std::vector<float> router_values;
  router_values.reserve(batch_size * seq_len * num_experts);
  for (int i = 0; i < batch_size * seq_len; ++i) {
    for (int j = 0; j < num_experts; ++j) {
      router_values.push_back(static_cast<float>(j) * 0.1f);
    }
  }
  // use custom logits and residual tensor for precision verification
  auto router_logits =
      CreateRouterLogits({batch_size * seq_len, num_experts}, router_values);
  auto residual = CreateCustomInput(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 100.0f));
  auto output =
      fused_moe->forward_expert(hidden_states, router_logits, residual);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  ASSERT_EQ(output.sizes().size(), 3) << "Output should be 3D tensor";
  ASSERT_EQ(output.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.size(1), seq_len) << "Sequence length should match";
  ASSERT_EQ(output.size(2), hidden_size) << "Hidden size should match";

  // Set expected output values for precision verification
  // TODO: Replace these placeholder values with your expected output
  // The expected values should be calculated based on your specific test case
  std::vector<float> expected_values;

  // Fill expected_values with placeholder data using custom dimensions
  expected_values.reserve(batch_size * seq_len * hidden_size);
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < seq_len; ++j) {
      for (int k = 0; k < hidden_size; ++k) {
        expected_values.push_back(1064.0f);  // Placeholder - to be calculated
      }
    }
  }

  SetExpectedOutput(expected_values);

  // Note: The precision verification is commented out until you set the
  // expected values. Uncomment the following line after setting the correct
  // expected values
  VerifyPrecision(output, 1e-3, 1e-4);
}

}  // namespace layer
}  // namespace xllm
