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

#include "model_input_factory.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "causal_lm.h"
#include "causal_vlm.h"
#include "dit_model.h"

namespace xllm {
namespace model_input {
namespace {

class FakeCausalLM final : public CausalLM {
 public:
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams&) override {
    return ModelOutput();
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) override {
    return torch::Tensor();
  }
  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_{torch::TensorOptions().device(torch::kCPU)};
};

class FakeCausalVLM final : public CausalVLM {
 public:
  MMDict encode(const ModelInputParams&) override { return MMDict(); }
  torch::Tensor get_input_embeddings(const torch::Tensor&,
                                     const ModelInputParams&) override {
    return torch::Tensor();
  }
  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams&) override {
    return ModelOutput();
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) override {
    return torch::Tensor();
  }
  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_{torch::TensorOptions().device(torch::kCPU)};
};

class FakeDiTModel final : public DiTModel {
 public:
  DiTForwardOutput forward(const DiTForwardInput&) override {
    return DiTForwardOutput();
  }
  torch::Device device() const override { return torch::Device(torch::kCPU); }
  const torch::TensorOptions& options() const override { return options_; }
  void load_model(std::unique_ptr<DiTModelLoader>) override {}

 private:
  torch::TensorOptions options_{torch::TensorOptions().device(torch::kCPU)};
};

TEST(ModelInputFactoryTest, CreateForLlmIncludesOnlyLlmByDefault) {
  FakeCausalLM model;
  ModelInputParams params;

  const ModelInput input = ModelInputFactory::create_for_llm(model, params);
  EXPECT_TRUE(input.has_llm());
  EXPECT_FALSE(input.has_vlm());
  EXPECT_FALSE(input.has_dit());
  EXPECT_FALSE(input.has_rec());
}

TEST(ModelInputFactoryTest, CreateForVlmIncludesVlmPartition) {
  FakeCausalVLM model;
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));

  const ModelInput input = ModelInputFactory::create_for_vlm(model, params);
  EXPECT_TRUE(input.has_llm());
  EXPECT_TRUE(input.has_vlm());
  EXPECT_FALSE(input.has_dit());
}

TEST(ModelInputFactoryTest, CreateForDitIncludesDitPartition) {
  FakeDiTModel model;
  ModelInputParams params;
  params.dit_forward_input.prompts.push_back("dit prompt");

  const ModelInput input = ModelInputFactory::create_for_dit(model, params);
  EXPECT_FALSE(input.has_llm());
  EXPECT_FALSE(input.has_vlm());
  EXPECT_TRUE(input.has_dit());
  EXPECT_FALSE(input.has_rec());
}

TEST(ModelInputFactoryTest, CreateForRecIncludesRecPartitionWhenPresent) {
  FakeCausalLM model;
  ModelInputParams params;
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  const ModelInput input = ModelInputFactory::create_for_rec(model, params);
  EXPECT_TRUE(input.has_llm());
  EXPECT_TRUE(input.has_rec());
}

TEST(ModelInputFactoryTest, CausalLmCreatesLlmInputDirectly) {
  FakeCausalLM model;
  ModelInputParams params;
  const ModelInput input = model.create_model_input(params);
  EXPECT_TRUE(input.has_llm());
  EXPECT_FALSE(input.has_vlm());
  EXPECT_FALSE(input.has_dit());
}

TEST(ModelInputFactoryTest, CausalVlmCreatesVlmInputDirectly) {
  FakeCausalVLM model;
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));
  const ModelInput input = model.create_model_input(params);
  EXPECT_TRUE(input.has_llm());
  EXPECT_TRUE(input.has_vlm());
  EXPECT_FALSE(input.has_dit());
}

TEST(ModelInputFactoryTest, DitModelCreatesDitInputDirectly) {
  FakeDiTModel model;
  ModelInputParams params;
  params.dit_forward_input.prompts.push_back("dit prompt");
  const ModelInput input = model.create_model_input(params);
  EXPECT_FALSE(input.has_llm());
  EXPECT_FALSE(input.has_vlm());
  EXPECT_TRUE(input.has_dit());
}

}  // namespace
}  // namespace model_input
}  // namespace xllm
