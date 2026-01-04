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
    model_args_ = test::create_default_model_args();

    // Initialize w8a8 quantization arguments
    quant_args_ = test::create_default_quant_args();

    // Initialize tensor options
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Device::type_torch(), 0)
                   .requires_grad(false);

    // Create mock ProcessGroup and initialize ParallelArgs
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);

    // Note: FusedMoE will be created by individual test cases with their
    // desired dimensions
  }

  void TearDown() override {
    // Clean up if needed
  }

  // Helper function to create router logits tensor
  torch::Tensor create_router_logits(const std::vector<int64_t>& shape,
                                     const std::vector<float>& values) {
    return test::create_custom_input(shape, values, options_);
  }

  std::unordered_map<std::string, torch::Tensor> create_default_test_weights(
      int64_t num_experts,
      int64_t hidden_size,
      int64_t intermediate_size) {
    // Create test weights for each expert
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
      std::string expert_prefix = "experts." + std::to_string(expert_id) + ".";

      // Create gate_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto gate_weight =
          torch::full({intermediate_size, hidden_size}, 2.0f, options_);
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
      auto up_weight =
          torch::full({intermediate_size, hidden_size}, 2.0f, options_);
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
          torch::full({hidden_size, intermediate_size}, 3.0f, options_);
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
    auto gate_weight = torch::full({num_experts, hidden_size}, 5.0f, options_);
    auto e_score_correction_bias = torch::full({num_experts}, 0.1f, options_);

    // Create shared experts weights
    auto shared_expert_up_weight =
        torch::full({intermediate_size, hidden_size}, 1.5f, options_);
    auto shared_expert_up_qweight = shared_expert_up_weight.to(torch::kInt8);
    auto shared_expert_up_scale = torch::full({intermediate_size},
                                              0.1f,
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(options_.device()));
    auto shared_expert_up_smooth = torch::full({hidden_size},
                                               0.05f,
                                               torch::TensorOptions()
                                                   .dtype(torch::kFloat32)
                                                   .device(options_.device()));
    auto shared_expert_down_weight =
        torch::full({hidden_size, intermediate_size}, 1.3f, options_);
    auto shared_expert_down_qweight =
        shared_expert_down_weight.to(torch::kInt8);
    auto shared_expert_down_scale = torch::full({hidden_size},
                                                0.1f,
                                                torch::TensorOptions()
                                                    .dtype(torch::kFloat32)
                                                    .device(options_.device()));
    auto shared_expert_down_smooth =
        torch::full({intermediate_size},
                    0.05f,
                    torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(options_.device()));

    // gate
    weight_dict["gate.weight"] = gate_weight;
    weight_dict["gate.e_score_correction_bias"] = e_score_correction_bias;

    // shared experts
    weight_dict["shared_experts.up_proj.qweight"] = shared_expert_up_qweight;
    weight_dict["shared_experts.up_proj.per_channel_scale"] =
        shared_expert_up_scale;
    weight_dict["shared_experts.up_proj.smooth"] = shared_expert_up_smooth;
    weight_dict["shared_experts.down_proj.qweight"] =
        shared_expert_down_qweight;
    weight_dict["shared_experts.down_proj.per_channel_scale"] =
        shared_expert_down_scale;
    weight_dict["shared_experts.down_proj.smooth"] = shared_expert_down_smooth;

    LOG(INFO) << "Test w8a8 smoothquant weights created successfully for "
              << num_experts << " experts";
    LOG(INFO) << "Hidden size: " << hidden_size
              << ", Intermediate size: " << intermediate_size;

    return weight_dict;
  }

  // Helper function to create FusedMoE with custom dimensions
  FusedMoE create_fused_moe(int64_t num_experts,
                            int64_t top_k,
                            int64_t num_expert_group,
                            int64_t topk_group,
                            double route_scale,
                            int64_t hidden_size,
                            int64_t intermediate_size,
                            int64_t n_shared_experts = 1,
                            bool is_gated = true,
                            bool has_score_bias = false,
                            bool has_bias = false,
                            bool skip_bias_add = false,
                            int64_t renormalize = 0,
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
  std::unordered_map<std::string, torch::Tensor> create_test_weights(
      int64_t num_experts,
      int64_t custom_hidden_size = -1,
      int64_t custom_intermediate_size = -1) {
    // Use custom sizes if provided, otherwise use model_args_ values
    int64_t test_hidden_size = (custom_hidden_size > 0)
                                   ? custom_hidden_size
                                   : model_args_.hidden_size();
    int64_t test_intermediate_size = (custom_intermediate_size > 0)
                                         ? custom_intermediate_size
                                         : model_args_.intermediate_size();

    return create_default_test_weights(
        num_experts, test_hidden_size, test_intermediate_size);
  }

  // Helper function to verify tensor values are close to expected
  void verify_tensor_close(const torch::Tensor& actual,
                           const torch::Tensor& expected,
                           double rtol = 1e-5,
                           double atol = 1e-8) {
    test::verify_tensor_close(actual, expected, rtol, atol);
  }

  // Helper function to create custom input tensor for precision testing
  torch::Tensor create_custom_input(const std::vector<int64_t>& shape,
                                    const std::vector<float>& values) {
    return test::create_custom_input(shape, values, options_);
  }

  // Helper function to set expected output for precision verification
  void set_expected_output(const std::vector<float>& expected_values) {
    expected_output_ = expected_values;
  }

  // Helper function to verify precision against expected output
  void verify_precision(const torch::Tensor& actual_output,
                        double rtol = 1e-3,
                        double atol = 1e-4) {
    test::verify_precision(actual_output, expected_output_, rtol, atol);
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
  const int64_t num_experts = 16;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;
  const bool gated = true;
  const bool has_score_bias = true;
  const bool has_bias = false;
  const bool skip_bias_add = false;
  const int64_t renormalize = 1;
  const int64_t n_shared_experts = 1;

  // Create FusedMoE with default dimensions
  auto fused_moe = create_fused_moe(num_experts,
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
      create_test_weights(num_experts, hidden_size, intermediate_size);

  // Load weights into the FusedMoE
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  // Create input tensors
  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  // Create router logits (batch_size * seq_len, num_experts)
  std::vector<float> router_values;
  router_values.reserve(batch_size * seq_len * num_experts);
  for (size_t i = 0; i < batch_size * seq_len; ++i) {
    for (size_t j = 0; j < num_experts; ++j) {
      router_values.push_back(static_cast<float>(j) * 0.1f);
    }
  }
  auto router_logits =
      create_router_logits({batch_size * seq_len, num_experts}, router_values);
  auto score_bias = torch::full({num_experts}, 0.1f, options_);
  auto output =
      fused_moe->forward_experts(hidden_states,
                                 router_logits,
                                 /*enable_all2all_communication=*/false);

  // Verify output shape
  CHECK_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  CHECK_EQ(output.size(0), batch_size * seq_len)
      << "The number of tokens should match";
  CHECK_EQ(output.size(1), hidden_size) << "The hidden size should match";

  // Verify output is not all zeros (weights were loaded)
  auto output_sum = torch::sum(output).item<float>();
  CHECK_NE(output_sum, 0.0f)
      << "Output should not be all zeros after loading weights";

  LOG(INFO) << "State dict loading test passed - output sum: " << output_sum;
}

TEST_F(FusedMoETest, PrecisionVerificationTest) {
  // Test loading weights into the FusedMoE
  const int64_t batch_size = 16;
  const int64_t seq_len = 32;
  const int64_t hidden_size = 7168;
  const int64_t intermediate_size = 2048;
  const int64_t num_experts = 16;
  const int64_t num_expert_group = 4;
  const int64_t topk_group = 4;
  const int64_t top_k = 2;
  const double route_scale = 2.5;
  const bool gated = true;
  const bool has_score_bias = true;
  const bool has_bias = false;
  const bool skip_bias_add = false;
  const int64_t renormalize = 1;
  const int64_t n_shared_experts = 1;

  // Create FusedMoE with default dimensions
  auto fused_moe = create_fused_moe(num_experts,
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
      create_test_weights(num_experts, hidden_size, intermediate_size);

  // Load weights into the FusedMoE
  StateDict state_dict(weight_dict);
  fused_moe->load_state_dict(state_dict);

  // Create input tensors
  auto hidden_states = create_custom_input(
      {batch_size * seq_len, hidden_size},
      std::vector<float>(batch_size * seq_len * hidden_size, 0.05f));

  // Create router logits (batch_size * seq_len, num_experts)
  std::vector<float> router_values;
  router_values.reserve(batch_size * seq_len * num_experts);
  for (size_t i = 0; i < batch_size * seq_len; ++i) {
    for (size_t j = 0; j < num_experts; ++j) {
      router_values.push_back(static_cast<float>(j) * 0.1f);
    }
  }
  // use custom logits and residual tensor for precision verification
  auto router_logits =
      create_router_logits({batch_size * seq_len, num_experts}, router_values);
  auto output =
      fused_moe->forward_experts(hidden_states,
                                 router_logits,
                                 /*enable_all2all_communication=*/false);

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  CHECK_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  CHECK_EQ(output.size(0), batch_size * seq_len)
      << "Batch size * seq_len should match";
  CHECK_EQ(output.size(1), hidden_size) << "Hidden size should match";

  // Set expected output values for precision verification
  // The expected values should be calculated based on your specific test case
  std::vector<float> expected_values;

  // Fill expected_values with placeholder data using custom dimensions
  expected_values.reserve(batch_size * seq_len * hidden_size);
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < seq_len; ++j) {
      for (size_t k = 0; k < hidden_size; ++k) {
        expected_values.push_back(1792.0f);  // calculated via vLLM MLU
      }
    }
  }

  set_expected_output(expected_values);

  // Note: The precision verification is commented out until you set the
  // expected values. Uncomment the following line after setting the correct
  // expected values
  verify_precision(output, 1e-3, 1e-4);
}

}  // namespace layer
}  // namespace xllm
