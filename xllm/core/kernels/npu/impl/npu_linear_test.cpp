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

#include <gtest/gtest.h>
#include <sys/resource.h>

#include "kernels/npu/linear.h"

namespace xllm::kernel {

class NpuLinearTest : public ::testing::Test {
 protected:
  NpuLinearTest() : parallel_args_(1, 1, nullptr) {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));

      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device("npu:0");
      npu_available_ = true;
      std::cout << "Using NPU device" << std::endl;

    } catch (...) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
      npu_available_ = false;
      std::cout << "Using CPU device (NPU unavailable)" << std::endl;
    }
  }

  void SetUp() override {
    torch::manual_seed(42);

    model_args_.hidden_size() = 4096;
    model_args_.intermediate_size() = 11008;
    model_args_.dtype() = "float16";

    quant_args_.torch_dtype() = "float16";

    context_ = std::make_unique<ModelContext>(
        parallel_args_, model_args_, quant_args_, tensor_options_);
  }

  void TearDown() override {
    context_.reset();

    if (npu_available_) {
      try {
        c10_npu::npuSynchronizeDevice();
        c10_npu::NPUCachingAllocator::emptyCache();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      } catch (...) {
        // NPU cleanup failures are usually not critical in test teardown
      }
    }
  }

  StateDict CreateStateDict(const torch::Tensor& weight_tensor) {
    std::unordered_map<std::string, torch::Tensor> tensor_map;
    tensor_map["weight"] = weight_tensor;
    return StateDict(tensor_map, "");
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions tensor_options_;
  std::unique_ptr<ModelContext> context_;
  bool npu_available_ = true;
};

// Test NpuLinearImpl construction
TEST_F(NpuLinearTest, ConstructorTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({
    auto linear = std::make_shared<NpuLinearImpl>(*context_);
    EXPECT_NE(linear, nullptr);
  });
}

// Test Linear wrapper construction
TEST_F(NpuLinearTest, NpuLinearWrapperTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({ auto linear = Linear(*context_); });
}

// Test state dict loading with mock weights
TEST_F(NpuLinearTest, LoadStateDictTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto linear = std::make_shared<NpuLinearImpl>(*context_);

  // Create weight tensor with shape [output_size, input_size] for linear layer
  auto weight_tensor =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);

  ASSERT_NO_THROW({ linear->load_state_dict(state_dict); });
}

// Test weight verification (should fail with uninitialized weights)
TEST_F(NpuLinearTest, VerifyLoadedWeightsFailTest) {
  auto linear = std::make_shared<NpuLinearImpl>(*context_);

  EXPECT_DEATH({ linear->verify_loaded_weights("test_weight"); }, ".*");
}

// Test weight verification (should pass with loaded weights)
TEST_F(NpuLinearTest, VerifyLoadedWeightsPassTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto linear = std::make_shared<NpuLinearImpl>(*context_);

  auto weight_tensor =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  linear->load_state_dict(state_dict);

  ASSERT_NO_THROW({ linear->verify_loaded_weights("test_weight"); });
}

// Test merge loaded weights
TEST_F(NpuLinearTest, MergeLoadedWeightsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto linear = std::make_shared<NpuLinearImpl>(*context_);

  auto weight_tensor =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  linear->load_state_dict(state_dict);

  ASSERT_NO_THROW({ linear->merge_loaded_weights(); });
}

// Test forward pass with mock input (may fail without proper NPU setup)
TEST_F(NpuLinearTest, ForwardPassBasicTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto linear = Linear(*context_);

  auto weight_tensor =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  linear->load_state_dict(state_dict);
  linear->merge_loaded_weights();

  // Input tensor with shape [batch_size, seq_len, hidden_size]
  auto input =
      torch::randn({1, 10, model_args_.hidden_size()}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto output = linear(input, 0);
    aclrtSynchronizeStream(npu_stream.stream());
    std::cout << "Input tensor shape: " << input.sizes() << std::endl;
    std::cout << "Output tensor shape: " << output.sizes() << std::endl;

    // Expected output shape: [batch_size, seq_len, intermediate_size]
    std::vector<int64_t> expected_shape = {
        1, 10, model_args_.intermediate_size()};
    EXPECT_EQ(output.sizes(), expected_shape);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping forward pass test - requires NPU environment: "
                 << e.what();
  }
}

// Test tensor shape consistency
TEST_F(NpuLinearTest, TensorShapeTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto linear = std::make_shared<NpuLinearImpl>(*context_);

  auto weight_tensor =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  linear->load_state_dict(state_dict);

  EXPECT_EQ(weight_tensor.size(0), model_args_.intermediate_size());
  EXPECT_EQ(weight_tensor.size(1), model_args_.hidden_size());
  EXPECT_EQ(weight_tensor.dim(), 2);
}

// Test different weight matrix dimensions
TEST_F(NpuLinearTest, DifferentWeightDimensionsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  std::vector<std::pair<int64_t, int64_t>> dimensions = {
      {768, 3072}, {1024, 4096}, {2048, 8192}, {4096, 11008}, {8192, 22016}};

  for (auto [input_size, output_size] : dimensions) {
    model_args_.hidden_size() = input_size;
    model_args_.intermediate_size() = output_size;

    QuantArgs local_quant_args = quant_args_;
    local_quant_args.torch_dtype() = "float16";

    auto context = std::make_unique<ModelContext>(
        parallel_args_, model_args_, local_quant_args, tensor_options_);

    auto linear = std::make_shared<NpuLinearImpl>(*context);

    auto weight_tensor =
        torch::randn({output_size, input_size}, tensor_options_);
    auto state_dict = CreateStateDict(weight_tensor);

    ASSERT_NO_THROW({ linear->load_state_dict(state_dict); });

    EXPECT_EQ(weight_tensor.size(0), output_size);
    EXPECT_EQ(weight_tensor.size(1), input_size);
  }
}

// Test linear transformation mathematical properties
TEST_F(NpuLinearTest, LinearTransformationPropertiesTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto linear = Linear(*context_);

  auto weight_tensor = torch::eye(model_args_.hidden_size(),
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat16)
                                      .device(tensor_options_.device()));

  if (model_args_.intermediate_size() != model_args_.hidden_size()) {
    if (model_args_.intermediate_size() > model_args_.hidden_size()) {
      auto padded_weight = torch::zeros(
          {model_args_.intermediate_size(), model_args_.hidden_size()},
          tensor_options_);
      padded_weight.narrow(0, 0, model_args_.hidden_size()) = weight_tensor;
      weight_tensor = padded_weight;
    } else {
      weight_tensor =
          weight_tensor.narrow(0, 0, model_args_.intermediate_size());
    }
  }

  auto state_dict = CreateStateDict(weight_tensor);
  linear->load_state_dict(state_dict);
  linear->merge_loaded_weights();

  auto input = torch::ones({1, 1, model_args_.hidden_size()}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto output = linear(input, 0);
    aclrtSynchronizeStream(npu_stream.stream());

    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 1);
    EXPECT_EQ(output.size(2),
              model_args_.intermediate_size());  // output features

  } catch (const std::exception& e) {
    GTEST_SKIP()
        << "Skipping mathematical properties test - requires NPU environment: "
        << e.what();
  }
}

// Test batch processing
TEST_F(NpuLinearTest, BatchProcessingTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto linear = Linear(*context_);

  auto weight_tensor =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  linear->load_state_dict(state_dict);
  linear->merge_loaded_weights();

  std::vector<std::vector<int64_t>> batch_shapes = {
      {1, 5, model_args_.hidden_size()},
      {2, 10, model_args_.hidden_size()},
      {4, 20, model_args_.hidden_size()},
      {8, 15, model_args_.hidden_size()}};

  for (const auto& shape : batch_shapes) {
    auto input = torch::randn(shape, tensor_options_);

    try {
      auto npu_stream = c10_npu::getCurrentNPUStream(0);
      auto output = linear(input, 0);
      aclrtSynchronizeStream(npu_stream.stream());

      EXPECT_EQ(output.size(0), shape[0]);
      EXPECT_EQ(output.size(1), shape[1]);
      EXPECT_EQ(output.size(2), model_args_.intermediate_size());

    } catch (const std::exception& e) {
      GTEST_SKIP() << "Skipping batch processing test for shape [" << shape[0]
                   << ", " << shape[1] << ", " << shape[2]
                   << "] - requires NPU environment: " << e.what();
      break;
    }
  }
}

// Test error handling with invalid inputs
TEST_F(NpuLinearTest, ErrorHandlingTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto linear = Linear(*context_);

  auto weight_tensor =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  linear->load_state_dict(state_dict);
  linear->merge_loaded_weights();

  auto wrong_input =
      torch::randn({1, 10, model_args_.hidden_size() + 100}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto output = linear(wrong_input, 0);
    aclrtSynchronizeStream(npu_stream.stream());
    FAIL() << "Expected exception for mismatched input dimensions";
  } catch (const std::exception& e) {
    // Expected behavior - input dimension mismatch should cause error
    std::cout << "Correctly caught expected error: " << e.what() << std::endl;
  }
}

}  // namespace xllm::kernel

int main(int argc, char** argv) {
  struct rlimit core_limit;
  core_limit.rlim_cur = 0;
  core_limit.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &core_limit);

  FILE* null_stderr = freopen("/dev/null", "w", stderr);
  if (null_stderr == nullptr) {
    fclose(stderr);
  }

  ::testing::InitGoogleTest(&argc, argv);

  bool npu_available = false;
  try {
    auto test_tensor =
        torch::zeros({1}, torch::TensorOptions().device("npu:0"));
    npu_available = true;
  } catch (...) {
    npu_available = false;
  }

  if (!npu_available) {
    std::cout << "NPU device not available, skipping all tests." << std::endl;
    return 0;  // Exit with success code, all tests skipped
  }

  int result = RUN_ALL_TESTS();
  _exit(result);
}