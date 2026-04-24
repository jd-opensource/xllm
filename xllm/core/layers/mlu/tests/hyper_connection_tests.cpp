/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cmath>
#include <string>
#include <unordered_map>

#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v4/hyper_connection.h"
#include "platform/device.h"

namespace xllm {
namespace layer {
namespace {

void expect_stats(const torch::Tensor& tensor,
                  double expected_sum,
                  double expected_min,
                  double expected_max) {
  EXPECT_TRUE(tensor.defined() && tensor.numel() > 0)
      << "Tensor must be defined and non-empty";
  torch::Tensor flat = tensor.flatten().to(torch::kFloat32);
  if (tensor.device() != torch::kCPU) {
    flat = flat.cpu();
  }

  double actual_sum = torch::sum(flat).item<double>();
  double actual_min = torch::min(flat).item<double>();
  double actual_max = torch::max(flat).item<double>();
  const double rtol = 1e-2;
  const double atol = 1e-5;

  auto within_tol = [rtol, atol](double actual, double expected) {
    double tol = atol + rtol * std::abs(expected);
    return std::abs(actual - expected) <= tol;
  };

  EXPECT_TRUE(within_tol(actual_sum, expected_sum))
      << "sum mismatch: actual=" << actual_sum << ", expected=" << expected_sum;
  EXPECT_TRUE(within_tol(actual_min, expected_min))
      << "min mismatch: actual=" << actual_min << ", expected=" << expected_min;
  EXPECT_TRUE(within_tol(actual_max, expected_max))
      << "max mismatch: actual=" << actual_max << ", expected=" << expected_max;
}

}  // namespace

class HyperConnectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Device::type_torch(), 0)
                   .requires_grad(false);
    options_fp32_ = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(Device::type_torch(), 0)
                        .requires_grad(false);
  }

  HyperConnectionHead create_hc_head(int64_t hc_mult,
                                     int64_t dim,
                                     float hc_eps,
                                     float norm_eps) {
    return HyperConnectionHead(
        HyperConnectionHeadImpl(hc_mult, dim, hc_eps, norm_eps, options_));
  }

  HyperConnectionPre create_hc_pre(int64_t hc_mult,
                                   int64_t dim,
                                   int64_t hc_sinkhorn_iters,
                                   float hc_eps,
                                   float norm_eps) {
    return HyperConnectionPre(HyperConnectionPreImpl(
        hc_mult, dim, hc_sinkhorn_iters, hc_eps, norm_eps, options_));
  }

  HyperConnectionPost create_hc_post() {
    return HyperConnectionPost(HyperConnectionPostImpl(options_));
  }

  std::unordered_map<std::string, torch::Tensor> create_hc_head_weights(
      int64_t hc_mult,
      int64_t dim,
      float fn_val,
      float base_val,
      float scale_val) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    int64_t hc_dim = hc_mult * dim;

    weight_dict["hc_head_fn"] =
        torch::full({hc_mult, hc_dim}, fn_val, options_fp32_);
    weight_dict["hc_head_base"] =
        torch::full({hc_mult}, base_val, options_fp32_);
    weight_dict["hc_head_scale"] = torch::full({1}, scale_val, options_fp32_);

    LOG(INFO) << "HyperConnectionHead weights created:";
    LOG(INFO) << "  hc_head_fn shape: " << weight_dict["hc_head_fn"].sizes();
    LOG(INFO) << "  hc_head_base shape: "
              << weight_dict["hc_head_base"].sizes();
    LOG(INFO) << "  hc_head_scale shape: "
              << weight_dict["hc_head_scale"].sizes();

    return weight_dict;
  }

  std::unordered_map<std::string, torch::Tensor> create_hc_pre_weights(
      int64_t hc_mult,
      int64_t dim,
      float fn_val,
      float base_val,
      float scale_val) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    int64_t mix_hc = (2 + hc_mult) * hc_mult;
    int64_t hc_dim = hc_mult * dim;
    weight_dict["hc_fn"] = torch::full({mix_hc, hc_dim}, fn_val, options_fp32_);
    weight_dict["hc_base"] = torch::full({mix_hc}, base_val, options_fp32_);
    weight_dict["hc_scale"] = torch::full({3}, scale_val, options_fp32_);

    LOG(INFO) << "HyperConnectionPre weights created:";
    LOG(INFO) << "  hc_fn shape: " << weight_dict["hc_fn"].sizes();
    LOG(INFO) << "  hc_base shape: " << weight_dict["hc_base"].sizes();
    LOG(INFO) << "  hc_scale shape: " << weight_dict["hc_scale"].sizes();

    return weight_dict;
  }

  void synchronize_device() {
    xllm::Device device(options_.device());
    device.synchronize_default_stream();
  }

  torch::TensorOptions options_;
  torch::TensorOptions options_fp32_;
};

TEST_F(HyperConnectionTest, HyperConnectionHeadForwardTest) {
  const int64_t batch_size = 2;
  const int64_t hc_mult = 4;
  const int64_t dim = 4096;
  const float hc_eps = 1e-6f;
  const float norm_eps = 1e-6f;

  const float fn_val = 0.1f;
  const float base_val = 0.2f;
  const float scale_val = 0.3f;

  HyperConnectionHead hc_head = create_hc_head(hc_mult, dim, hc_eps, norm_eps);
  std::unordered_map<std::string, torch::Tensor> weight_dict =
      create_hc_head_weights(hc_mult, dim, fn_val, base_val, scale_val);
  hc_head->load_state_dict(StateDict(weight_dict));

  torch::Tensor hidden_states =
      torch::full({batch_size, hc_mult, dim}, 0.1f, options_);

  LOG(INFO) << "Testing HyperConnectionHead forward pass";
  LOG(INFO) << "  Input shape: " << hidden_states.sizes();
  torch::Tensor output = hc_head->forward(hidden_states);

  synchronize_device();

  LOG(INFO) << "  Output shape: " << output.sizes();
  ASSERT_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.size(1), dim) << "Output dim should be dim";

  expect_stats(output, 3280.0f, 0.400390625f, 0.400390625f);

  LOG(INFO) << "HyperConnectionHead forward test passed";
}

TEST_F(HyperConnectionTest, HyperConnectionPreForwardTest) {
  const int64_t batch_size = 2;
  const int64_t hc_mult = 4;
  const int64_t dim = 4096;
  const int64_t hc_sinkhorn_iters = 20;
  const float hc_eps = 1e-6f;
  const float norm_eps = 1e-6f;

  const float fn_val = 0.1f;
  const float base_val = 0.2f;
  const float scale_val = 0.3f;

  HyperConnectionPre hc_pre =
      create_hc_pre(hc_mult, dim, hc_sinkhorn_iters, hc_eps, norm_eps);
  std::unordered_map<std::string, torch::Tensor> weight_dict =
      create_hc_pre_weights(hc_mult, dim, fn_val, base_val, scale_val);
  hc_pre->load_state_dict(StateDict(weight_dict));

  torch::Tensor hidden_states =
      torch::full({batch_size, hc_mult, dim}, 0.1f, options_);

  LOG(INFO) << "Testing HyperConnectionPre forward pass";
  LOG(INFO) << "  Input shape: " << hidden_states.sizes();
  PrePostCombOutput output = hc_pre->forward(hidden_states);

  synchronize_device();

  LOG(INFO) << "  Pre output shape: " << output.pre.sizes();
  ASSERT_EQ(output.pre.sizes().size(), 2) << "Pre output should be 2D tensor";
  ASSERT_EQ(output.pre.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.pre.size(1), dim) << "Dim should match";

  LOG(INFO) << "  Post output shape: " << output.post.sizes();
  ASSERT_EQ(output.post.sizes().size(), 2) << "Post output should be 2D tensor";
  ASSERT_EQ(output.post.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.post.size(1), hc_mult) << "Post dim should be hc_mult";

  LOG(INFO) << "  Comb output shape: " << output.comb.sizes();
  ASSERT_EQ(output.comb.sizes().size(), 3) << "Comb output should be 3D tensor";
  ASSERT_EQ(output.comb.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.comb.size(1), hc_mult) << "Comb dim1 should be hc_mult";
  ASSERT_EQ(output.comb.size(2), hc_mult) << "Comb dim2 should be hc_mult";

  LOG(INFO) << "Verifying pre output:";
  expect_stats(output.pre, 3280.0f, 0.400390625f, 0.400390625f);

  LOG(INFO) << "Verifying post output:";
  expect_stats(output.post, 16.0f, 2.0f, 2.0f);

  LOG(INFO) << "Verifying comb output:";
  expect_stats(output.comb, 8.0f, 0.25f, 0.25f);

  LOG(INFO) << "HyperConnectionPre forward test passed";
}

TEST_F(HyperConnectionTest, HyperConnectionPostForwardTest) {
  const int64_t batch_size = 2;
  const int64_t hc_mult = 4;
  const int64_t dim = 4096;
  const int64_t hc_sinkhorn_iters = 20;
  const float hc_eps = 1e-6f;
  const float norm_eps = 1e-6f;

  const float fn_val = 0.1f;
  const float base_val = 0.2f;
  const float scale_val = 0.3f;

  HyperConnectionPre hc_pre =
      create_hc_pre(hc_mult, dim, hc_sinkhorn_iters, hc_eps, norm_eps);
  std::unordered_map<std::string, torch::Tensor> weight_dict =
      create_hc_pre_weights(hc_mult, dim, fn_val, base_val, scale_val);
  hc_pre->load_state_dict(StateDict(weight_dict));

  torch::Tensor h = torch::full({batch_size, hc_mult, dim}, 0.1f, options_);
  PrePostCombOutput pre_output = hc_pre->forward(h);

  synchronize_device();

  HyperConnectionPost hc_post = create_hc_post();

  LOG(INFO) << "Testing HyperConnectionPost forward pass";
  LOG(INFO) << "  x (pre) shape: " << pre_output.pre.sizes();
  LOG(INFO) << "  residual (h) shape: " << h.sizes();
  LOG(INFO) << "  post shape: " << pre_output.post.sizes();
  LOG(INFO) << "  comb shape: " << pre_output.comb.sizes();

  torch::Tensor output =
      hc_post->forward(pre_output.pre, h, pre_output.post, pre_output.comb);

  synchronize_device();

  LOG(INFO) << "  Output shape: " << output.sizes();
  ASSERT_EQ(output.sizes().size(), 3) << "Output should be 3D tensor";
  ASSERT_EQ(output.size(0), batch_size) << "Batch size should match";
  ASSERT_EQ(output.size(1), hc_mult) << "hc_mult should match";
  ASSERT_EQ(output.size(2), dim) << "Dim should match";

  expect_stats(output, 29568.0f, 0.90234375f, 0.90234375f);

  LOG(INFO) << "HyperConnectionPost forward test passed";
}

}  // namespace layer
}  // namespace xllm
