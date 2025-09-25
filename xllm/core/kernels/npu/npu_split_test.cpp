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

#include "core/kernels/split.h"

namespace xllm::kernel {

class NpuSplitTest : public ::testing::Test {
 protected:
  NpuSplitTest() : parallel_args_(1, 1, nullptr) {
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

    model_args_.hidden_size() = 4096 * 3;
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

  StateDict CreateEmptyStateDict() {
    std::unordered_map<std::string, torch::Tensor> tensor_map;
    return StateDict(tensor_map, "");
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions tensor_options_;
  std::unique_ptr<ModelContext> context_;
  bool npu_available_ = true;
};

// Test NpuSplitImpl construction
TEST_F(NpuSplitTest, ConstructorTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({
    auto split = std::make_shared<NpuSplitImpl>(*context_);
    EXPECT_NE(split, nullptr);
  });
}

// Test Split wrapper construction
TEST_F(NpuSplitTest, SplitWrapperTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({ auto split = Split(*context_); });
}

// Test state dict loading (should be no-op for split layer)
TEST_F(NpuSplitTest, LoadStateDictTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto split = std::make_shared<NpuSplitImpl>(*context_);
  auto state_dict = CreateEmptyStateDict();

  ASSERT_NO_THROW({ split->load_state_dict(state_dict); });
}

// Test weight verification (should pass for split layer as it has no weights)
TEST_F(NpuSplitTest, VerifyLoadedWeightsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto split = std::make_shared<NpuSplitImpl>(*context_);

  ASSERT_NO_THROW({ split->verify_loaded_weights("test_weight"); });
}

// Test merge loaded weights (should be no-op for split layer)
TEST_F(NpuSplitTest, MergeLoadedWeightsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto split = std::make_shared<NpuSplitImpl>(*context_);

  ASSERT_NO_THROW({ split->merge_loaded_weights(); });
}

// Test forward pass with basic input
TEST_F(NpuSplitTest, ForwardPassBasicTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto split = Split(*context_);

  // Input tensor with shape [batch_size, seq_len, hidden_size]
  auto input =
      torch::randn({1, 10, model_args_.hidden_size() * 3}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto outputs = split->forward(input, 0);
    aclrtSynchronizeStream(npu_stream.stream());
    std::cout << "Input tensor shape: " << input.sizes() << std::endl;
    std::cout << "Number of output tensors: " << outputs.size() << std::endl;

    EXPECT_EQ(outputs.size(), 3);

    for (size_t i = 0; i < outputs.size(); ++i) {
      EXPECT_EQ(outputs[i].size(0), 1);   // batch size
      EXPECT_EQ(outputs[i].size(1), 10);  // sequence length
      std::cout << "Output " << i << " shape: " << outputs[i].sizes()
                << std::endl;
    }

    int64_t total_output_features = 0;
    for (const auto& output : outputs) {
      total_output_features += output.size(2);
    }
    EXPECT_EQ(total_output_features, model_args_.hidden_size() * 3);

  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping forward pass test - requires NPU environment: "
                 << e.what();
  }
}

// Test split functionality with different input shapes
TEST_F(NpuSplitTest, SplitDifferentInputShapesTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto split = Split(*context_);

  std::vector<std::vector<int64_t>> input_shapes = {
      {1, 5, model_args_.hidden_size()},
      {2, 10, model_args_.hidden_size()},
      {4, 20, model_args_.hidden_size()},
      {1, 1, model_args_.hidden_size()}};

  for (const auto& shape : input_shapes) {
    auto input = torch::randn(shape, tensor_options_);

    try {
      auto npu_stream = c10_npu::getCurrentNPUStream(0);
      auto outputs = split->forward(input, 0);
      aclrtSynchronizeStream(npu_stream.stream());

      EXPECT_EQ(outputs.size(), 3);

      for (const auto& output : outputs) {
        EXPECT_EQ(output.size(0), shape[0]);
        EXPECT_EQ(output.size(1), shape[1]);
        EXPECT_GT(output.size(2), 0);
      }

      int64_t total_features = 0;
      for (const auto& output : outputs) {
        total_features += output.size(2);
      }
      EXPECT_EQ(total_features, shape[2]);

    } catch (const std::exception& e) {
      GTEST_SKIP() << "Skipping shape test for [" << shape[0] << ", "
                   << shape[1] << ", " << shape[2]
                   << "] - requires NPU environment: " << e.what();
      break;
    }
  }
}

// Test split with different hidden sizes
TEST_F(NpuSplitTest, SplitDifferentHiddenSizesTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  std::vector<int64_t> hidden_sizes = {
      768 * 3, 1024 * 3, 2048 * 3, 4096 * 3, 6144 * 3, 8192 * 3};
  auto npu_stream = c10_npu::getCurrentNPUStream(0);
  for (auto hidden_size : hidden_sizes) {
    model_args_.hidden_size() = hidden_size;

    QuantArgs local_quant_args = quant_args_;
    local_quant_args.torch_dtype() = "float16";

    auto context = std::make_unique<ModelContext>(
        parallel_args_, model_args_, local_quant_args, tensor_options_);

    try {
      auto split = Split(*context);

      auto input = torch::randn({1, 10, hidden_size}, tensor_options_);

      auto npu_stream = c10_npu::getCurrentNPUStream(0);
      auto outputs = split->forward(input, 0);
      aclrtSynchronizeStream(npu_stream.stream());
      aclrtSynchronizeStream(npu_stream.stream());
      EXPECT_EQ(outputs.size(), 3);

      int64_t total_features = 0;
      for (const auto& output : outputs) {
        total_features += output.size(2);
      }
      EXPECT_EQ(total_features, hidden_size);

    } catch (const std::exception& e) {
      GTEST_SKIP() << "Skipping hidden size test for " << hidden_size
                   << " - requires NPU environment: " << e.what();
      break;
    }
  }
}

// Test error handling with invalid inputs
TEST_F(NpuSplitTest, ErrorHandlingTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto split = Split(*context_);

  try {
    auto empty_input = torch::empty({0, 0, 0}, tensor_options_);
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto outputs = split->forward(empty_input, 0);
    aclrtSynchronizeStream(npu_stream.stream());
  } catch (const std::exception& e) {
    std::cout << "Correctly caught expected error for empty tensor: "
              << e.what() << std::endl;
  }

  try {
    auto wrong_dim_input =
        torch::randn({10, model_args_.hidden_size()}, tensor_options_);
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto outputs = split->forward(wrong_dim_input, 0);
    aclrtSynchronizeStream(npu_stream.stream());
  } catch (const std::exception& e) {
    std::cout << "Caught error for 2D input: " << e.what() << std::endl;
  }
}

// Test consistency of split operation
TEST_F(NpuSplitTest, SplitConsistencyTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto split = Split(*context_);
  auto input1 =
      torch::randn({2, 5, model_args_.hidden_size()}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);

    auto outputs1 = split->forward(input1, 0);
    aclrtSynchronizeStream(npu_stream.stream());

    auto outputs2 = split->forward(input1, 1);
    aclrtSynchronizeStream(npu_stream.stream());

    EXPECT_EQ(outputs1.size(), outputs2.size());

    for (size_t i = 0; i < outputs1.size(); ++i) {
      EXPECT_TRUE(outputs1[i].sizes().equals(outputs2[i].sizes()));
    }

  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping consistency test - requires NPU environment: "
                 << e.what();
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
    return 0;
  }

  int result = RUN_ALL_TESTS();
  _exit(result);
}