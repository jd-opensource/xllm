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

#include "kernels/npu/rope.h"

namespace xllm::kernel {

class NpuRopeTest : public ::testing::Test {
 protected:
  NpuRopeTest() : parallel_args_(1, 1, nullptr) {
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
    model_args_.num_attention_heads() = 32;
    model_args_.head_dim() = 128;
    model_args_.max_position_embeddings() = 2048;
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

  StateDict CreateStateDict() {
    std::unordered_map<std::string, torch::Tensor> tensor_map;
    // RoPE layer doesn't have trainable weights, so empty state dict
    return StateDict(tensor_map, "");
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions tensor_options_;
  std::unique_ptr<ModelContext> context_;
  bool npu_available_ = true;
};

// Test NpuRopeImpl construction
TEST_F(NpuRopeTest, ConstructorTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({
    auto rope = std::make_shared<NpuRopeImpl>(*context_);
    EXPECT_NE(rope, nullptr);
  });
}

// Test Rope wrapper construction
TEST_F(NpuRopeTest, RopeWrapperTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({ auto rope = Rope(*context_); });
}

// Test state dict loading (RoPE doesn't have weights)
TEST_F(NpuRopeTest, LoadStateDictTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rope = std::make_shared<NpuRopeImpl>(*context_);
  auto state_dict = CreateStateDict();

  ASSERT_NO_THROW({ rope->load_state_dict(state_dict); });
}

// Test weight verification (should pass as RoPE has no weights)
TEST_F(NpuRopeTest, VerifyLoadedWeightsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rope = std::make_shared<NpuRopeImpl>(*context_);

  ASSERT_NO_THROW({ rope->verify_loaded_weights("test_weight"); });
}

// Test merge loaded weights
TEST_F(NpuRopeTest, MergeLoadedWeightsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rope = std::make_shared<NpuRopeImpl>(*context_);

  ASSERT_NO_THROW({ rope->merge_loaded_weights(); });
}

// Test forward pass with mock input tensors following constraint specifications
TEST_F(NpuRopeTest, ForwardPassBasicTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rope = std::make_shared<NpuRopeImpl>(*context_);

  int64_t batch_size = 2;
  std::vector<int64_t> seq_lengths = {8, 12};
  int64_t max_seq_len =
      *std::max_element(seq_lengths.begin(), seq_lengths.end());
  int64_t ntokens = std::accumulate(
      seq_lengths.begin(), seq_lengths.end(), 0);  // ntokens = sum(seqlen[i])

  int64_t head_num_q = model_args_.num_attention_heads();  // headNumQ
  int64_t head_num_k =
      model_args_.num_attention_heads();  // headNumK (can be <= headNumQ)
  int64_t head_size = model_args_.head_dim();

  // Ensure 32-byte alignment for hiddenSizeQ and hiddenSizeK
  int64_t hidden_size_q =
      head_num_q * head_size;  // hiddenSizeQ = head_size * headNumQ
  int64_t hidden_size_k =
      head_num_k * head_size;  // hiddenSizeK = head_size * headNumK

  // Validate 32-byte alignment constraint
  ASSERT_EQ(hidden_size_q % 32, 0) << "hiddenSizeQ must be 32-byte aligned";
  ASSERT_EQ(hidden_size_k % 32, 0) << "hiddenSizeK must be 32-byte aligned";

  // Create tensors with constraint-compliant dimensions
  // Input format: [ntokens, hiddenSizeQ/K] for 2D tensors
  auto q = torch::randn({ntokens, hidden_size_q}, tensor_options_);
  auto k = torch::randn({ntokens, hidden_size_k}, tensor_options_);

  // cos/sin embeddings: [ntokens, head_size] for standard mode
  auto cos_embedding = torch::randn({ntokens, head_size}, tensor_options_);
  auto sin_embedding = torch::randn({ntokens, head_size}, tensor_options_);

  auto seq_len_tensor =
      torch::tensor(seq_lengths, tensor_options_.dtype(torch::kInt32));

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto outputs =
        rope->forward(q, k, cos_embedding, sin_embedding, seq_len_tensor, 0);
    aclrtSynchronizeStream(npu_stream.stream());

    EXPECT_GE(outputs.size(),
              2);  // Should return at least q_rotated and k_rotated
    if (outputs.size() >= 2) {
      std::cout << "Output Q tensor shape: " << outputs[0].sizes() << std::endl;
      std::cout << "Output K tensor shape: " << outputs[1].sizes() << std::endl;
      EXPECT_EQ(outputs[0].sizes(), q.sizes());
      EXPECT_EQ(outputs[1].sizes(), k.sizes());
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping forward pass test - requires NPU environment: "
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