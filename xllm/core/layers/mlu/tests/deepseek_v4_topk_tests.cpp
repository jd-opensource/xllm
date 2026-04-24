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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <unordered_map>
#include <utility>

#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_topk.h"

namespace xllm {
namespace layer {

namespace {

constexpr int64_t kRoutedExperts = 6;
constexpr int64_t kActivatedExperts = 2;
constexpr int64_t kVocabSize = 8;

DeepSeekV4TopK create_topk(float route_scale) {
  torch::TensorOptions options = torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(torch::kCPU)
                                     .requires_grad(false);
  return DeepSeekV4TopK(DeepSeekV4TopKImpl(kRoutedExperts,
                                           kActivatedExperts,
                                           route_scale,
                                           kVocabSize,
                                           /*use_hash=*/false,
                                           options));
}

void load_bias(DeepSeekV4TopK& topk, const torch::Tensor& bias) {
  std::unordered_map<std::string, torch::Tensor> weights;
  weights["bias"] = bias.to(torch::kFloat32);
  StateDict state_dict(std::move(weights));
  topk->load_state_dict(state_dict);
}

}  // namespace

TEST(DeepSeekV4TopKTest, ForwardReturnsExpectedShape) {
  DeepSeekV4TopK topk = create_topk(/*route_scale=*/1.0f);
  load_bias(topk, torch::zeros({kRoutedExperts}));

  torch::Tensor scores = torch::tensor({{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f},
                                        {0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f},
                                        {0.0f, 0.3f, 0.6f, 0.9f, 1.2f, 1.5f}},
                                       torch::kFloat32);

  DeepSeekV4TopKOutput output = topk->forward(scores);

  ASSERT_TRUE(output.weights.defined());
  ASSERT_TRUE(output.indices.defined());
  EXPECT_EQ(output.weights.dim(), 2);
  EXPECT_EQ(output.indices.dim(), 2);
  EXPECT_EQ(output.weights.size(0), scores.size(0));
  EXPECT_EQ(output.weights.size(1), kActivatedExperts);
  EXPECT_EQ(output.indices.size(0), scores.size(0));
  EXPECT_EQ(output.indices.size(1), kActivatedExperts);
  EXPECT_EQ(output.indices.dtype(), torch::kInt32);
}

TEST(DeepSeekV4TopKTest, TopKIndicesStayInExpertRange) {
  DeepSeekV4TopK topk = create_topk(/*route_scale=*/1.0f);
  load_bias(topk, torch::zeros({kRoutedExperts}));

  torch::manual_seed(2026);
  torch::Tensor scores = torch::randn({5, kRoutedExperts}, torch::kFloat32);
  DeepSeekV4TopKOutput output = topk->forward(scores);
  torch::Tensor indices_i64 = output.indices.to(torch::kInt64);

  EXPECT_GE(indices_i64.min().item<int64_t>(), 0);
  EXPECT_LT(indices_i64.max().item<int64_t>(), kRoutedExperts);
  EXPECT_TRUE(torch::allclose(output.weights.sum(-1),
                              torch::ones({scores.size(0)}, torch::kFloat32)));
}

TEST(DeepSeekV4TopKTest, LoadedBiasMakesRoutingDeterministic) {
  DeepSeekV4TopK topk = create_topk(/*route_scale=*/2.0f);
  load_bias(topk, torch::tensor({0.0f, 10.0f, 1.0f, 9.0f, 2.0f, 8.0f}));

  torch::Tensor scores = torch::zeros({3, kRoutedExperts}, torch::kFloat32);
  DeepSeekV4TopKOutput output = topk->forward(scores);
  torch::Tensor expected_indices =
      torch::tensor({{1, 3}, {1, 3}, {1, 3}}, torch::kInt32);
  torch::Tensor expected_weights =
      torch::ones({scores.size(0), kActivatedExperts}, torch::kFloat32);

  EXPECT_TRUE(torch::equal(output.indices, expected_indices));
  EXPECT_TRUE(torch::allclose(output.weights, expected_weights));
}

TEST(DeepSeekV4TopKTest, LoadedHashTableSelectsExpectedExperts) {
  torch::TensorOptions options = torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(torch::kCPU)
                                     .requires_grad(false);
  DeepSeekV4TopK topk = DeepSeekV4TopK(DeepSeekV4TopKImpl(kRoutedExperts,
                                                          kActivatedExperts,
                                                          /*route_scale=*/1.0f,
                                                          kVocabSize,
                                                          /*use_hash=*/true,
                                                          options));
  torch::Tensor tid2eid = torch::tensor(
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 0}, {0, 2}, {1, 3}},
      torch::kInt32);
  std::unordered_map<std::string, torch::Tensor> weights;
  weights["tid2eid"] = tid2eid;
  StateDict state_dict(std::move(weights));
  topk->load_state_dict(state_dict);

  torch::Tensor scores = torch::ones({3, kRoutedExperts}, torch::kFloat32);
  torch::Tensor input_ids = torch::tensor({0, 3, 5}, torch::kInt64);
  DeepSeekV4TopKOutput output = topk->forward(scores, input_ids);
  torch::Tensor expected_indices =
      torch::tensor({{0, 1}, {3, 4}, {5, 0}}, torch::kInt32);

  EXPECT_TRUE(torch::equal(output.indices, expected_indices));
  EXPECT_TRUE(torch::allclose(output.weights,
                              torch::full({3, kActivatedExperts}, 0.5f)));
}

}  // namespace layer
}  // namespace xllm
