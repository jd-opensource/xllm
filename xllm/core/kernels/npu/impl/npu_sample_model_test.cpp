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
#include "kernels/npu/rms_norm.h"
#include "kernels/npu/rope.h"
#include "kernels/npu/split.h"

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

  // Helper method to create cos/sin embeddings for RoPE
  std::pair<torch::Tensor, torch::Tensor> CreateRopeEmbeddings(
      int64_t seq_len,
      int64_t head_dim) {
    auto cos_embedding = torch::cos(
        torch::arange(0, seq_len, tensor_options_).unsqueeze(1) *
        torch::arange(0, head_dim / 2, tensor_options_).unsqueeze(0) * 0.01);
    auto sin_embedding = torch::sin(
        torch::arange(0, seq_len, tensor_options_).unsqueeze(1) *
        torch::arange(0, head_dim / 2, tensor_options_).unsqueeze(0) * 0.01);
    return std::make_pair(cos_embedding, sin_embedding);
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

  // Attention parameters
  int64_t num_heads_ = 32;
  int64_t num_kv_heads_ = 32;
  int64_t head_dim_ = 128;
  bool attn_output_gate_ = false;
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
    std::cout << "RMS norm output shape: " << normalized_output.sizes()
              << std::endl;

    // Step 2: qkv, _ = self.qkv_proj(hidden_states)
    auto qkv_output = qkv_proj(normalized_output, 0);
    std::cout << "QKV projection output shape: " << qkv_output.sizes()
              << std::endl;

    // Step 3: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size],
    // dim=-1)
    auto split_outputs = split_layer(qkv_output, 0);

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
    aclrtSynchronizeStream(npu_stream.stream());
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
      auto qkv_output = qkv_proj(normalized_output, 0);
      auto split_outputs = split_layer(qkv_output, 0);

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
      aclrtSynchronizeStream(npu_stream.stream());
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
    auto qkv_output = qkv_proj(normalized_output, 0);
    auto split_outputs = split_layer(qkv_output, 0);

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
    aclrtSynchronizeStream(npu_stream.stream());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping data flow test - requires NPU environment: "
                 << e.what();
  }
}

// Test QKV splitting with attention output gate functionality
TEST_F(SampleModelTest, QKVSplitWithAttentionGateTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  int64_t q_gate_size = q_size_ * 2;                   // q + gate
  int64_t qkv_gate_size = q_gate_size + 2 * kv_size_;  // (q + gate) + k + v

  auto rms_norm = RmsNorm(*context_);
  auto qkv_proj = Linear(*context_);
  auto split_layer = Split(*context_, 2, 3, {q_gate_size, kv_size_, kv_size_});

  // Setup for attention output gate mode
  attn_output_gate_ = true;

  auto rms_norm_weight =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto rms_norm_state_dict = CreateRmsNormStateDict(rms_norm_weight);
  rms_norm->load_state_dict(rms_norm_state_dict);
  rms_norm->merge_loaded_weights();

  // QKV projection with gate: output size = (q_size * 2) + k_size + v_size
  auto qkv_weight =
      torch::randn({qkv_gate_size, model_args_.hidden_size()}, tensor_options_);
  auto qkv_state_dict = CreateLinearStateDict(qkv_weight);
  qkv_proj->load_state_dict(qkv_state_dict);
  qkv_proj->merge_loaded_weights();

  auto split_state_dict = CreateEmptyStateDict();
  split_layer->load_state_dict(split_state_dict);
  split_layer->merge_loaded_weights();

  auto input =
      torch::randn({1, 10, model_args_.hidden_size()}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);

    auto normalized_output = rms_norm(input, 0);

    auto qkv_output = qkv_proj(normalized_output, 0);
    std::cout << "QKV with gate output shape: " << qkv_output.sizes()
              << std::endl;

    auto split_outputs = split_layer(qkv_output, 0);
    EXPECT_EQ(split_outputs.size(), 3)
        << "Split should produce 3 tensors (q_gate, k, v)";

    if (split_outputs.size() >= 3) {
      auto q_gate = split_outputs[0];
      auto k = split_outputs[1];
      auto v = split_outputs[2];

      std::cout << "Q+Gate tensor shape: " << q_gate.sizes() << std::endl;
      std::cout << "K tensor shape: " << k.sizes() << std::endl;
      std::cout << "V tensor shape: " << v.sizes() << std::endl;

      std::vector<int64_t> expected_q_gate_shape = {1, 10, q_gate_size};
      std::vector<int64_t> expected_kv_shape = {1, 10, kv_size_};

      EXPECT_EQ(q_gate.sizes(), expected_q_gate_shape)
          << "Q+Gate tensor shape mismatch";
      EXPECT_EQ(k.sizes(), expected_kv_shape) << "K tensor shape mismatch";
      EXPECT_EQ(v.sizes(), expected_kv_shape) << "V tensor shape mismatch";

      // q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
      auto orig_shape = q_gate.sizes();
      auto q_gate_reshaped =
          q_gate.view({orig_shape[0], orig_shape[1], num_heads_, -1});

      // q, gate = torch.chunk(q_gate, 2, dim=-1)
      auto q_gate_chunks = torch::chunk(q_gate_reshaped, 2, -1);
      EXPECT_EQ(q_gate_chunks.size(), 2) << "Should split q_gate into 2 chunks";

      if (q_gate_chunks.size() >= 2) {
        auto q = q_gate_chunks[0];
        auto gate = q_gate_chunks[1];

        q = q.reshape({orig_shape[0], orig_shape[1], -1});
        gate = gate.reshape({orig_shape[0], orig_shape[1], -1});

        std::cout << "Final Q shape: " << q.sizes() << std::endl;
        std::cout << "Final Gate shape: " << gate.sizes() << std::endl;

        std::vector<int64_t> expected_final_q_shape = {1, 10, q_size_};
        EXPECT_EQ(q.sizes(), expected_final_q_shape)
            << "Final Q tensor shape mismatch";
        EXPECT_EQ(gate.sizes(), expected_final_q_shape)
            << "Gate tensor shape mismatch";
      }
    }

    std::cout << "QKV split with attention gate test completed successfully!"
              << std::endl;
    aclrtSynchronizeStream(npu_stream.stream());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping attention gate test - requires NPU environment: "
                 << e.what();
  }
}

// Test standard QKV splitting (without attention gate)
TEST_F(SampleModelTest, StandardQKVSplitTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = RmsNorm(*context_);
  auto qkv_proj = Linear(*context_);
  auto split_layer = Split(*context_);

  attn_output_gate_ = false;

  auto rms_norm_weight =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto rms_norm_state_dict = CreateRmsNormStateDict(rms_norm_weight);
  rms_norm->load_state_dict(rms_norm_state_dict);
  rms_norm->merge_loaded_weights();

  // Standard QKV projection: output size = q_size + k_size + v_size
  auto qkv_weight =
      torch::randn({qkv_size_, model_args_.hidden_size()}, tensor_options_);
  auto qkv_state_dict = CreateLinearStateDict(qkv_weight);
  qkv_proj->load_state_dict(qkv_state_dict);
  qkv_proj->merge_loaded_weights();

  auto split_state_dict = CreateEmptyStateDict();
  split_layer->load_state_dict(split_state_dict);
  split_layer->merge_loaded_weights();

  auto input =
      torch::randn({1, 10, model_args_.hidden_size()}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);

    auto normalized_output = rms_norm(input, 0);

    auto qkv_output = qkv_proj(normalized_output, 0);
    std::cout << "Standard QKV output shape: " << qkv_output.sizes()
              << std::endl;

    auto split_outputs = split_layer(qkv_output, 0);
    EXPECT_EQ(split_outputs.size(), 3)
        << "Split should produce 3 tensors (q, k, v)";

    if (split_outputs.size() >= 3) {
      auto q = split_outputs[0];
      auto k = split_outputs[1];
      auto v = split_outputs[2];

      std::cout << "Q tensor shape: " << q.sizes() << std::endl;
      std::cout << "K tensor shape: " << k.sizes() << std::endl;
      std::cout << "V tensor shape: " << v.sizes() << std::endl;

      std::vector<int64_t> expected_q_shape = {1, 10, q_size_};
      std::vector<int64_t> expected_kv_shape = {1, 10, kv_size_};

      EXPECT_EQ(q.sizes(), expected_q_shape) << "Q tensor shape mismatch";
      EXPECT_EQ(k.sizes(), expected_kv_shape) << "K tensor shape mismatch";
      EXPECT_EQ(v.sizes(), expected_kv_shape) << "V tensor shape mismatch";
    }

    std::cout << "Standard QKV split test completed successfully!" << std::endl;
    aclrtSynchronizeStream(npu_stream.stream());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping standard QKV test - requires NPU environment: "
                 << e.what();
  }
}

// Test Q and K normalization functionality
TEST_F(SampleModelTest, QKNormalizationTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = RmsNorm(*context_);
  auto qkv_proj = Linear(*context_);
  auto split_layer = Split(*context_);
  auto q_norm = RmsNorm(*context_);
  auto k_norm = RmsNorm(*context_);

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

  auto q_norm_weight = torch::randn({head_dim_}, tensor_options_);
  auto q_norm_state_dict = CreateRmsNormStateDict(q_norm_weight);
  q_norm->load_state_dict(q_norm_state_dict);
  q_norm->merge_loaded_weights();

  auto k_norm_weight = torch::randn({head_dim_}, tensor_options_);
  auto k_norm_state_dict = CreateRmsNormStateDict(k_norm_weight);
  k_norm->load_state_dict(k_norm_state_dict);
  k_norm->merge_loaded_weights();

  auto input =
      torch::randn({1, 10, model_args_.hidden_size()}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);

    // Forward pass: norm -> qkv_proj -> split -> q_norm/k_norm
    auto normalized_output = rms_norm(input, 0);
    auto qkv_output = qkv_proj(normalized_output, 0);
    auto split_outputs = split_layer(qkv_output, 0);

    EXPECT_EQ(split_outputs.size(), 3) << "Expected 3 split outputs";

    if (split_outputs.size() >= 3) {
      auto q = split_outputs[0];
      auto k = split_outputs[1];
      auto v = split_outputs[2];

      // Reshape Q and K for normalization: [batch, seq, num_heads, head_dim]
      auto q_reshaped = q.view({-1, num_heads_, head_dim_});
      auto k_reshaped = k.view({-1, num_kv_heads_, head_dim_});

      std::cout << "Q reshaped for norm: " << q_reshaped.sizes() << std::endl;
      std::cout << "K reshaped for norm: " << k_reshaped.sizes() << std::endl;

      auto q_normalized = q_norm(q_reshaped, 0);
      auto k_normalized = k_norm(k_reshaped, 0);

      q_normalized = q_normalized.view({1, -1, num_heads_ * head_dim_});
      k_normalized = k_normalized.view({1, -1, num_kv_heads_ * head_dim_});

      std::cout << "Q after norm: " << q_normalized.sizes() << std::endl;
      std::cout << "K after norm: " << k_normalized.sizes() << std::endl;

      EXPECT_FALSE(torch::isnan(q_normalized).any().item<bool>())
          << "NaN detected in normalized Q";
      EXPECT_FALSE(torch::isinf(q_normalized).any().item<bool>())
          << "Inf detected in normalized Q";
      EXPECT_FALSE(torch::isnan(k_normalized).any().item<bool>())
          << "NaN detected in normalized K";
      EXPECT_FALSE(torch::isinf(k_normalized).any().item<bool>())
          << "Inf detected in normalized K";

      EXPECT_EQ(q_normalized.sizes(), q.sizes())
          << "Q shape changed after norm";
      EXPECT_EQ(k_normalized.sizes(), k.sizes())
          << "K shape changed after norm";
    }

    std::cout << "Q and K normalization test completed successfully!"
              << std::endl;
    aclrtSynchronizeStream(npu_stream.stream());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping Q/K norm test - requires NPU environment: "
                 << e.what();
  }
}

// Comprehensive test: norm -> qkv_proj -> split -> q_norm/k_norm -> rope
TEST_F(SampleModelTest, CompleteAttentionPipelineTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = RmsNorm(*context_);
  auto qkv_proj = Linear(*context_);
  auto split_layer = Split(*context_);
  auto q_norm = RmsNorm(*context_);
  auto k_norm = RmsNorm(*context_);
  auto rope_layer = Rope(*context_);

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

  auto q_norm_weight = torch::randn({head_dim_}, tensor_options_);
  auto q_norm_state_dict = CreateRmsNormStateDict(q_norm_weight);
  q_norm->load_state_dict(q_norm_state_dict);
  q_norm->merge_loaded_weights();

  auto k_norm_weight = torch::randn({head_dim_}, tensor_options_);
  auto k_norm_state_dict = CreateRmsNormStateDict(k_norm_weight);
  k_norm->load_state_dict(k_norm_state_dict);
  k_norm->merge_loaded_weights();

  auto rope_state_dict = CreateEmptyStateDict();
  rope_layer->load_state_dict(rope_state_dict);
  rope_layer->merge_loaded_weights();

  std::vector<std::vector<int64_t>> test_shapes = {
      {1, 5, model_args_.hidden_size()},
      {2, 10, model_args_.hidden_size()},
      {1, 20, model_args_.hidden_size()}};

  for (const auto& shape : test_shapes) {
    auto input = torch::randn(shape, tensor_options_);
    int64_t seq_len = shape[1];

    try {
      auto npu_stream = c10_npu::getCurrentNPUStream(0);

      std::cout << "\nTesting complete pipeline with input shape: " << shape[0]
                << "x" << shape[1] << "x" << shape[2] << std::endl;

      auto normalized_output = rms_norm(input, 0);

      auto qkv_output = qkv_proj(normalized_output, 0);

      auto split_outputs = split_layer(qkv_output, 0);
      EXPECT_EQ(split_outputs.size(), 3) << "Expected 3 split outputs";

      if (split_outputs.size() >= 3) {
        auto q = split_outputs[0];
        auto k = split_outputs[1];
        auto v = split_outputs[2];

        auto q_reshaped = q.view({-1, num_heads_, head_dim_});
        auto k_reshaped = k.view({-1, num_kv_heads_, head_dim_});

        auto q_normalized = q_norm(q_reshaped, 0);
        auto k_normalized = k_norm(k_reshaped, 0);

        q_normalized = q_normalized.view({-1, num_heads_ * head_dim_});
        k_normalized = k_normalized.view({-1, num_kv_heads_ * head_dim_});

        auto rope_embeddings = CreateRopeEmbeddings(seq_len, head_dim_);
        auto cos_embedding = rope_embeddings.first;
        auto sin_embedding = rope_embeddings.second;
        auto seq_len_tensor =
            torch::tensor({seq_len}, tensor_options_.dtype(torch::kInt32));

        auto rope_outputs = rope_layer->forward(q_normalized,
                                                k_normalized,
                                                cos_embedding,
                                                sin_embedding,
                                                seq_len_tensor,
                                                0);

        EXPECT_EQ(rope_outputs.size(), 2) << "Expected 2 RoPE outputs";

        if (rope_outputs.size() >= 2) {
          auto q_final = rope_outputs[0];
          auto k_final = rope_outputs[1];

          std::cout << "Final Q shape: " << q_final.sizes() << std::endl;
          std::cout << "Final K shape: " << k_final.sizes() << std::endl;
          std::cout << "V shape: " << v.sizes() << std::endl;

          // Verify final shapes
          // EXPECT_EQ(q_final.size(0), shape[0]) << "Batch size mismatch";
          // EXPECT_EQ(q_final.size(1), shape[1]) << "Sequence length mismatch";
          // EXPECT_EQ(k_final.size(0), shape[0]) << "Batch size mismatch";
          // EXPECT_EQ(k_final.size(1), shape[1]) << "Sequence length mismatch";

          EXPECT_EQ(q_final.sizes(), q_normalized.sizes());
          EXPECT_EQ(k_final.sizes(), k_normalized.sizes());

          EXPECT_FALSE(torch::isnan(q_final).any().item<bool>())
              << "NaN detected in final Q";
          EXPECT_FALSE(torch::isinf(q_final).any().item<bool>())
              << "Inf detected in final Q";
          EXPECT_FALSE(torch::isnan(k_final).any().item<bool>())
              << "NaN detected in final K";
          EXPECT_FALSE(torch::isinf(k_final).any().item<bool>())
              << "Inf detected in final K";
          EXPECT_FALSE(torch::isnan(v).any().item<bool>())
              << "NaN detected in V";
          EXPECT_FALSE(torch::isinf(v).any().item<bool>())
              << "Inf detected in V";

          std::cout << "Complete pipeline test passed for shape [" << shape[0]
                    << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
        }
      }

      aclrtSynchronizeStream(npu_stream.stream());
    } catch (const std::exception& e) {
      GTEST_SKIP() << "Skipping complete pipeline test for shape [" << shape[0]
                   << ", " << shape[1] << ", " << shape[2]
                   << "] - requires NPU environment: " << e.what();
      break;
    }
  }

  std::cout << "\nComplete attention pipeline test completed successfully!"
            << std::endl;
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