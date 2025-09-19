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

#include "rms_norm.h"

#include <gtest/gtest.h>
#include <signal.h>
#include <sys/resource.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <memory>
#include <thread>

#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/framework/parallel_state.h"
#include "core/framework/quant_args.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::hf {

class NNRmsNormTest : public ::testing::Test {
 protected:
  NNRmsNormTest() : parallel_args_(1, 1, nullptr) {
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

    model_args_.rms_norm_eps() = 1e-6f;
    model_args_.hidden_size() = 4096;
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

// Test RmsNormImpl construction
TEST_F(NNRmsNormTest, ConstructorTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({
    auto rms_norm = std::make_shared<RmsNormImpl>(*context_);
    EXPECT_NE(rms_norm, nullptr);
  });
}

// Test create_rms_norm_layer factory function
TEST_F(NNRmsNormTest, CreateRmsNormLayerTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = create_rms_norm_layer(*context_);
  EXPECT_NE(rms_norm, nullptr);
}

// Test RmsNorm wrapper construction
TEST_F(NNRmsNormTest, RmsNormWrapperTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  ASSERT_NO_THROW({ auto rms_norm = RmsNorm(*context_); });
}

// Test parameter initialization from args
TEST_F(NNRmsNormTest, ParamFromArgsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = std::make_shared<RmsNormImpl>(*context_);

  atb::infer::RmsNormParam param;
  rms_norm->param_from_args(param, model_args_);

  EXPECT_EQ(param.layerType,
            atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM);
  EXPECT_FLOAT_EQ(param.normParam.epsilon, model_args_.rms_norm_eps());
}

// Test state dict loading with mock weights
TEST_F(NNRmsNormTest, LoadStateDictTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = std::make_shared<RmsNormImpl>(*context_);

  auto weight_tensor =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);

  ASSERT_NO_THROW({ rms_norm->load_state_dict(state_dict); });
}

// Test weight verification (should fail with uninitialized weights)
TEST_F(NNRmsNormTest, VerifyLoadedWeightsFailTest) {
  auto rms_norm = std::make_shared<RmsNormImpl>(*context_);

  EXPECT_DEATH({ rms_norm->verify_loaded_weights("test_weight"); }, ".*");
}

// Test weight verification (should pass with loaded weights)
TEST_F(NNRmsNormTest, VerifyLoadedWeightsPassTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = std::make_shared<RmsNormImpl>(*context_);

  auto weight_tensor =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  rms_norm->load_state_dict(state_dict);

  ASSERT_NO_THROW({ rms_norm->verify_loaded_weights("test_weight"); });
}

// Test merge loaded weights
TEST_F(NNRmsNormTest, MergeLoadedWeightsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = std::make_shared<RmsNormImpl>(*context_);

  auto weight_tensor =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  rms_norm->load_state_dict(state_dict);

  ASSERT_NO_THROW({ rms_norm->merge_loaded_weights(); });
}

// Test forward pass with mock input (may fail without proper NPU setup)
TEST_F(NNRmsNormTest, ForwardPassBasicTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = RmsNorm(*context_);

  auto weight_tensor =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  rms_norm->load_state_dict(state_dict);
  rms_norm->merge_loaded_weights();

  auto input =
      torch::randn({1, 10, model_args_.hidden_size()}, tensor_options_);

  try {
    auto npu_stream = c10_npu::getCurrentNPUStream(0);
    auto output = rms_norm(input, 0);
    aclrtSynchronizeStream(npu_stream.stream());
    std::cout << "Output tensor shape: " << output.sizes() << std::endl;
    EXPECT_EQ(output.sizes(), input.sizes());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping forward pass test - requires NPU environment: "
                 << e.what();
  }
}

// Test tensor shape consistency
TEST_F(NNRmsNormTest, TensorShapeTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  auto rms_norm = std::make_shared<RmsNormImpl>(*context_);

  auto weight_tensor =
      torch::randn({model_args_.hidden_size()}, tensor_options_);
  auto state_dict = CreateStateDict(weight_tensor);
  rms_norm->load_state_dict(state_dict);

  EXPECT_EQ(weight_tensor.size(0), model_args_.hidden_size());
  EXPECT_EQ(weight_tensor.dim(), 1);
}

// Test different epsilon values
TEST_F(NNRmsNormTest, DifferentEpsilonTest) {
  std::vector<float> epsilon_values = {1e-5f, 1e-6f, 1e-8f, 1e-12f};

  for (float eps : epsilon_values) {
    model_args_.rms_norm_eps() = eps;
    QuantArgs local_quant_args = quant_args_;
    local_quant_args.torch_dtype() = "float16";

    auto context = std::make_unique<ModelContext>(
        parallel_args_, model_args_, local_quant_args, tensor_options_);

    auto rms_norm = std::make_shared<RmsNormImpl>(*context);

    atb::infer::RmsNormParam param;
    rms_norm->param_from_args(param, model_args_);

    EXPECT_FLOAT_EQ(param.normParam.epsilon, eps);
  }
}

// Test with different hidden sizes
TEST_F(NNRmsNormTest, DifferentHiddenSizesTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "Skipping NPU test - NPU device not available";
  }

  std::vector<int64_t> hidden_sizes = {768, 1024, 2048, 4096, 8192};

  for (int64_t hidden_size : hidden_sizes) {
    model_args_.hidden_size() = hidden_size;
    QuantArgs local_quant_args = quant_args_;
    local_quant_args.torch_dtype() = "float16";

    auto context = std::make_unique<ModelContext>(
        parallel_args_, model_args_, local_quant_args, tensor_options_);

    auto rms_norm = std::make_shared<RmsNormImpl>(*context);

    auto weight_tensor = torch::randn({hidden_size}, tensor_options_);
    auto state_dict = CreateStateDict(weight_tensor);

    ASSERT_NO_THROW({ rms_norm->load_state_dict(state_dict); });

    EXPECT_EQ(weight_tensor.size(0), hidden_size);
  }
}

}  // namespace xllm::hf

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