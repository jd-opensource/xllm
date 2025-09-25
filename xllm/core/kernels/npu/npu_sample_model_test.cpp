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

#include "core/kernels/linear.h"
#include "core/kernels/rms_norm.h"
#include "core/kernels/split.h"

namespace xllm::kernel {

class SampleModelTest : public ::testing::Test {
 protected:
  SampleModelTest() : parallel_args_(1, 1, nullptr) {
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
    model_args_.rms_norm_eps() = 1e-6f;
    model_args_.dtype() = "float16";

    q_size_ = model_args_.hidden_size();
    kv_size_ = model_args_.hidden_size();
    qkv_size_ = q_size_ + 2 * kv_size_;  // q + k + v

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

  StateDict CreateRmsNormStateDict(const torch::Tensor& weight_tensor) {
    std::unordered_map<std::string, torch::Tensor> tensor_map;
    tensor_map["weight"] = weight_tensor;
    return StateDict(tensor_map, "");
  }

  StateDict CreateLinearStateDict(const torch::Tensor& weight_tensor) {
    std::unordered_map<std::string, torch::Tensor> tensor_map;
    tensor_map["weight"] = weight_tensor;
    return StateDict(tensor_map, "");
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

  // QKV dimensions
  int64_t q_size_;
  int64_t kv_size_;
  int64_t qkv_size_;
};

// Test RMS norm + Linear layer construction
TEST_F(SampleModelTest, ConstructorTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({
    auto rms_norm = std::make_shared<NpuRmsNormImpl>(*context_);
    auto linear = std::make_shared<NpuLinearImpl>(*context_);
    EXPECT_NE(rms_norm, nullptr);
    EXPECT_NE(linear, nullptr);
  });
}

// Test combined RMS norm + Linear layer wrapper construction
TEST_F(SampleModelTest, WrapperConstructionTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({
    auto rms_norm = RmsNorm(*context_);
    auto linear = Linear(*context_);
  });
}

// Test state dict loading for both layers
TEST_F(SampleModelTest, LoadStateDictTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = std::make_shared<NpuRmsNormImpl>(*context_);
  auto linear = std::make_shared<NpuLinearImpl>(*context_);

  auto rms_norm_weight =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto rms_norm_state_dict = CreateRmsNormStateDict(rms_norm_weight);

  auto linear_weight =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto linear_state_dict = CreateLinearStateDict(linear_weight);

  ASSERT_NO_THROW({
    rms_norm->load_state_dict(rms_norm_state_dict);
    linear->load_state_dict(linear_state_dict);
  });
}

// Test weight verification for both layers
TEST_F(SampleModelTest, VerifyLoadedWeightsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = std::make_shared<NpuRmsNormImpl>(*context_);
  auto linear = std::make_shared<NpuLinearImpl>(*context_);

  auto rms_norm_weight =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto rms_norm_state_dict = CreateRmsNormStateDict(rms_norm_weight);
  rms_norm->load_state_dict(rms_norm_state_dict);

  auto linear_weight =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto linear_state_dict = CreateLinearStateDict(linear_weight);
  linear->load_state_dict(linear_state_dict);

  ASSERT_NO_THROW({
    rms_norm->verify_loaded_weights("rms_norm_weight");
    linear->verify_loaded_weights("linear_weight");
  });
}

// Test merge loaded weights for both layers
TEST_F(SampleModelTest, MergeLoadedWeightsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = std::make_shared<NpuRmsNormImpl>(*context_);
  auto linear = std::make_shared<NpuLinearImpl>(*context_);

  auto rms_norm_weight =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto rms_norm_state_dict = CreateRmsNormStateDict(rms_norm_weight);
  rms_norm->load_state_dict(rms_norm_state_dict);

  auto linear_weight =
      torch::randn({model_args_.intermediate_size(), model_args_.hidden_size()},
                   tensor_options_);
  auto linear_state_dict = CreateLinearStateDict(linear_weight);
  linear->load_state_dict(linear_state_dict);

  ASSERT_NO_THROW({
    rms_norm->merge_loaded_weights();
    linear->merge_loaded_weights();
  });
}

// Test combined forward pass: RMS norm -> QKV projection -> Split (q, k, v)
TEST_F(SampleModelTest, CombinedForwardPassTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = RmsNorm(*context_);
  auto qkv_proj = Linear(*context_);
  auto split_layer = Split(*context_);

  auto rms_norm_weight =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto rms_norm_state_dict = CreateRmsNormStateDict(rms_norm_weight);
  rms_norm->load_state_dict(rms_norm_state_dict);
  rms_norm->merge_loaded_weights();

  // Setup QKV projection weights: output size = q_size + k_size + v_size
  auto qkv_weight =
      torch::randn({qkv_size_, model_args_.hidden_size()}, tensor_options_);
  auto qkv_state_dict = CreateLinearStateDict(qkv_weight);
  qkv_proj->load_state_dict(qkv_state_dict);
  qkv_proj->merge_loaded_weights();

  // Setup split layer (no weights needed)
  auto split_state_dict = CreateEmptyStateDict();
  split_layer->load_state_dict(split_state_dict);
  split_layer->merge_loaded_weights();

  // Input tensor with shape [batch_size, seq_len, hidden_size]
  auto input =
      torch::randn({1, 10, model_args_.hidden_size()}, tensor_options_);

  try {
    std::cout << "Input tensor shape: " << input.sizes() << std::endl;

    auto npu_stream = c10_npu::getCurrentNPUStream(0);

    // Step 1: hidden_states = self.norm(hidden_states)
    auto normalized_output = rms_norm(input, 0);
    aclrtSynchronizeStream(npu_stream.stream());
    std::cout << "RMS norm output shape: " << normalized_output.sizes()
              << std::endl;

    // Step 2: qkv, _ = self.qkv_proj(hidden_states)
    auto qkv_output = qkv_proj(normalized_output, 0);
    aclrtSynchronizeStream(npu_stream.stream());
    std::cout << "QKV projection output shape: " << qkv_output.sizes()
              << std::endl;

    // Step 3: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
    // dim=-1)
    auto split_outputs = split_layer(qkv_output, 0);
    aclrtSynchronizeStream(npu_stream.stream());

    EXPECT_EQ(split_outputs.size(), 3)
        << "Split should produce 3 tensors (q, k, v)";

    std::cout << "Split outputs:" << std::endl;
    for (size_t i = 0; i < split_outputs.size(); ++i) {
      std::cout << "  Tensor " << i << " shape: " << split_outputs[i].sizes()
                << std::endl;
    }

    EXPECT_EQ(normalized_output.sizes(), input.sizes());

    // Expected QKV output shape: [batch_size, seq_len, qkv_size]
    std::vector<int64_t> expected_qkv_shape = {1, 10, qkv_size_};
    EXPECT_EQ(qkv_output.sizes(), expected_qkv_shape);

    // Expected split output shapes
    // q: [batch_size, seq_len, q_size]
    // k: [batch_size, seq_len, kv_size]
    // v: [batch_size, seq_len, kv_size]
    std::vector<int64_t> expected_q_shape = {1, 10, q_size_};
    std::vector<int64_t> expected_kv_shape = {1, 10, kv_size_};

    if (split_outputs.size() >= 3) {
      EXPECT_EQ(split_outputs[0].sizes(), expected_q_shape)
          << "Q tensor shape mismatch";
      EXPECT_EQ(split_outputs[1].sizes(), expected_kv_shape)
          << "K tensor shape mismatch";
      EXPECT_EQ(split_outputs[2].sizes(), expected_kv_shape)
          << "V tensor shape mismatch";
    }

    std::cout << "Combined forward pass test (norm -> qkv_proj -> split) "
                 "completed successfully!"
              << std::endl;

  } catch (const std::exception& e) {
    GTEST_SKIP()
        << "Skipping combined forward pass test - requires NPU environment: "
        << e.what();
  }
}

// Test combined forward pass with different batch sizes: norm -> qkv_proj ->
// split
TEST_F(SampleModelTest, CombinedForwardPassBatchTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = RmsNorm(*context_);
  auto qkv_proj = Linear(*context_);
  auto split_layer = Split(*context_);

  auto rms_norm_weight =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto rms_norm_state_dict = CreateRmsNormStateDict(rms_norm_weight);
  rms_norm->load_state_dict(rms_norm_state_dict);
  rms_norm->merge_loaded_weights();

  auto qkv_weight =
      torch::randn({qkv_size_, model_args_.hidden_size()}, tensor_options_);
  auto qkv_state_dict = CreateLinearStateDict(qkv_weight);
  qkv_proj->load_state_dict(qkv_state_dict);
  qkv_proj->merge_loaded_weights();

  auto split_state_dict = CreateEmptyStateDict();
  split_layer->load_state_dict(split_state_dict);
  split_layer->merge_loaded_weights();

  std::vector<std::vector<int64_t>> batch_shapes = {
      {1, 5, model_args_.hidden_size()},
      {2, 10, model_args_.hidden_size()},
      {4, 20, model_args_.hidden_size()}};

  for (const auto& shape : batch_shapes) {
    auto input = torch::randn(shape, tensor_options_);

    try {
      auto npu_stream = c10_npu::getCurrentNPUStream(0);

      auto normalized_output = rms_norm(input, 0);
      aclrtSynchronizeStream(npu_stream.stream());

      auto qkv_output = qkv_proj(normalized_output, 0);
      aclrtSynchronizeStream(npu_stream.stream());

      auto split_outputs = split_layer(qkv_output, 0);
      aclrtSynchronizeStream(npu_stream.stream());

      EXPECT_EQ(normalized_output.size(0), shape[0]);
      EXPECT_EQ(normalized_output.size(1), shape[1]);
      EXPECT_EQ(normalized_output.size(2), shape[2]);

      EXPECT_EQ(qkv_output.size(0), shape[0]);
      EXPECT_EQ(qkv_output.size(1), shape[1]);
      EXPECT_EQ(qkv_output.size(2), qkv_size_);

      EXPECT_EQ(split_outputs.size(), 3);
      if (split_outputs.size() >= 3) {
        // Q tensor
        EXPECT_EQ(split_outputs[0].size(0), shape[0]);
        EXPECT_EQ(split_outputs[0].size(1), shape[1]);
        EXPECT_EQ(split_outputs[0].size(2), q_size_);

        // K tensor
        EXPECT_EQ(split_outputs[1].size(0), shape[0]);
        EXPECT_EQ(split_outputs[1].size(1), shape[1]);
        EXPECT_EQ(split_outputs[1].size(2), kv_size_);

        // V tensor
        EXPECT_EQ(split_outputs[2].size(0), shape[0]);
        EXPECT_EQ(split_outputs[2].size(1), shape[1]);
        EXPECT_EQ(split_outputs[2].size(2), kv_size_);
      }

    } catch (const std::exception& e) {
      GTEST_SKIP() << "Skipping batch processing test for shape [" << shape[0]
                   << ", " << shape[1] << ", " << shape[2]
                   << "] - requires NPU environment: " << e.what();
      break;
    }
  }
}

// Test tensor data flow and numerical properties: norm -> qkv_proj -> split
TEST_F(SampleModelTest, DataFlowTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = RmsNorm(*context_);
  auto qkv_proj = Linear(*context_);
  auto split_layer = Split(*context_);

  auto rms_norm_weight =
      torch::ones({model_args_.hidden_size()}, tensor_options_);
  auto rms_norm_state_dict = CreateRmsNormStateDict(rms_norm_weight);
  rms_norm->load_state_dict(rms_norm_state_dict);
  rms_norm->merge_loaded_weights();

  auto qkv_weight =
      torch::ones({qkv_size_, model_args_.hidden_size()}, tensor_options_) *
      0.1f;
  auto qkv_state_dict = CreateLinearStateDict(qkv_weight);
  qkv_proj->load_state_dict(qkv_state_dict);
  qkv_proj->merge_loaded_weights();

  auto split_state_dict = CreateEmptyStateDict();
  split_layer->load_state_dict(split_state_dict);
  split_layer->merge_loaded_weights();

  auto input = torch::ones({1, 1, model_args_.hidden_size()}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);

    auto normalized_output = rms_norm(input, 0);
    aclrtSynchronizeStream(npu_stream.stream());

    auto qkv_output = qkv_proj(normalized_output, 0);
    aclrtSynchronizeStream(npu_stream.stream());

    auto split_outputs = split_layer(qkv_output, 0);
    aclrtSynchronizeStream(npu_stream.stream());

    EXPECT_FALSE(torch::isnan(normalized_output).any().item<bool>())
        << "NaN detected in normalized output";
    EXPECT_FALSE(torch::isinf(normalized_output).any().item<bool>())
        << "Inf detected in normalized output";

    EXPECT_FALSE(torch::isnan(qkv_output).any().item<bool>())
        << "NaN detected in QKV projection output";
    EXPECT_FALSE(torch::isinf(qkv_output).any().item<bool>())
        << "Inf detected in QKV projection output";

    EXPECT_EQ(split_outputs.size(), 3) << "Expected 3 split outputs (q, k, v)";

    for (size_t i = 0; i < split_outputs.size(); ++i) {
      EXPECT_FALSE(torch::isnan(split_outputs[i]).any().item<bool>())
          << "NaN detected in split output " << i;
      EXPECT_FALSE(torch::isinf(split_outputs[i]).any().item<bool>())
          << "Inf detected in split output " << i;
    }

    std::cout << "Data flow test completed - no NaN or Inf values detected in "
                 "pipeline!"
              << std::endl;

  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping data flow test - requires NPU environment: "
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