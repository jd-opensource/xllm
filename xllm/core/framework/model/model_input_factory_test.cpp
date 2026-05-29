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
#include <utility>
#include <vector>

#include "causal_lm.h"
#include "causal_vlm.h"
#include "dit_model.h"
#include "rec_causal_lm.h"

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

class FakeRecCausalLM final : public RecCausalLM {
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
  EXPECT_TRUE(has_llm(input));
  EXPECT_FALSE(has_vlm(input));
  EXPECT_FALSE(has_dit(input));
  EXPECT_FALSE(has_rec(input));
}

TEST(ModelInputFactoryTest, CreateForLlmIgnoresRecPartitionWhenPresent) {
  FakeCausalLM model;
  ModelInputParams params;
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  const ModelInput input = ModelInputFactory::create_for_llm(model, params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_FALSE(has_rec(input));
}

TEST(ModelInputFactoryTest, CreateForVlmIncludesVlmPartition) {
  FakeCausalVLM model;
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));

  const ModelInput input = ModelInputFactory::create_for_vlm(model, params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_vlm(input));
  EXPECT_FALSE(has_dit(input));
}

TEST(ModelInputFactoryTest, CreateForDitIncludesDitPartition) {
  FakeDiTModel model;
  ModelInputParams params;
  params.dit_forward_input.prompts.push_back("dit prompt");

  const ModelInput input = ModelInputFactory::create_for_dit(model, params);
  EXPECT_FALSE(has_llm(input));
  EXPECT_FALSE(has_vlm(input));
  EXPECT_TRUE(has_dit(input));
  EXPECT_FALSE(has_rec(input));
}

TEST(ModelInputFactoryTest, CreateForRecIncludesRecPartitionWhenPresent) {
  FakeCausalLM model;
  ModelInputParams params;
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  const ModelInput input = ModelInputFactory::create_for_rec(model, params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_rec(input));
}

TEST(ModelInputFactoryTest, CreateForRecKeepsRecPartitionEmptyWhenMissing) {
  FakeCausalLM model;
  ModelInputParams params;

  const ModelInput input = ModelInputFactory::create_for_rec(model, params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_FALSE(has_rec(input));
}

TEST(ModelInputFactoryTest, CausalLmCreatesLlmInputDirectly) {
  FakeCausalLM model;
  ModelInputParams params;
  const ModelInput input = model.create_model_input(params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_FALSE(has_vlm(input));
  EXPECT_FALSE(has_dit(input));
}

TEST(ModelInputFactoryTest, RecCausalLmCreatesRecInputDirectlyWhenPresent) {
  FakeRecCausalLM model;
  ModelInputParams params;
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  const ModelInput input = model.create_model_input(params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_rec(input));
}

TEST(ModelInputFactoryTest, CausalVlmCreatesVlmInputDirectly) {
  FakeCausalVLM model;
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));
  const ModelInput input = model.create_model_input(params);
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_vlm(input));
  EXPECT_FALSE(has_dit(input));
}

TEST(ModelInputFactoryTest, DitModelCreatesDitInputDirectly) {
  FakeDiTModel model;
  ModelInputParams params;
  params.dit_forward_input.prompts.push_back("dit prompt");
  const ModelInput input = model.create_model_input(params);
  EXPECT_FALSE(has_llm(input));
  EXPECT_FALSE(has_vlm(input));
  EXPECT_TRUE(has_dit(input));
}

TEST(ModelInputFactoryTest, MakeModelInputFromLegacyMoveKeepsPartitions) {
  ModelInputParams params;
  params.deep_stacks.push_back(torch::zeros({1, 1}));
  auto& onerec = params.mutable_onerec_params();
  onerec.bs = 2;

  const ModelInput input = make_model_input_from_legacy(std::move(params));
  EXPECT_TRUE(has_llm(input));
  EXPECT_TRUE(has_vlm(input));
  EXPECT_TRUE(has_rec(input));
}

}  // namespace
}  // namespace model_input
}  // namespace xllm
