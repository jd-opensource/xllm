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

#include <framework/core/device.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

#include "base_executor_impl.h"
#include "core/framework/batch/batch.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_output.h"
#include "mlu_graph_executor_impl.h"
#include "platform/device.h"
#include "runtime/options.h"

namespace xllm {
class MockCausalLM : public CausalLM {
 public:
  MockCausalLM(const torch::TensorOptions& options) : options_(options) {
    auto weight = torch::randn({1024, 1024}, options_) * 0.02;
    weight_ = register_parameter("weight", weight, false);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    auto hidden_states = params.input_embedding.matmul(weight_);
    return ModelOutput(hidden_states);
  }
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return hidden_states;
  }
  void load_model(std::unique_ptr<ModelLoader> loader) override {}
  torch::Device device() const override { return options_.device(); }
  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {}
  void update_expert_weight(int32_t layer_id) override {}
  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::Tensor input_;
  torch::Tensor weight_;
  torch::TensorOptions options_;
};

class MluGraphExecutorTest : public ::testing::Test {
 protected:
  MluGraphExecutorTest() = default;

  void SetUp() override {
    torch::Device device("mlu:0");
    tensor_options_ = torch::TensorOptions(torch::kBFloat16).device(device);

    model_args_.model_type("test_model");
    model_args_.dtype("bfloat16");
    model_args_.hidden_size(1024);
    model_args_.max_position_embeddings(2048);

    const uint32_t block_size = 16;
    options_.num_decoding_tokens(1);
    options_.block_size(block_size);

    model_ = std::make_unique<MockCausalLM>(tensor_options_);
    impl_ = std::make_unique<::xllm::mlu::MluGraphExecutorImpl>(
        model_.get(), model_args_, device, options_);
    base_impl_ = std::make_unique<BaseExecutorImpl>(
        model_.get(), model_args_, device, options_);
  }

  ForwardInput prepare_inputs(int32_t batch_size, uint64_t seed) {
    Device device(tensor_options_.device());
    device.set_seed(seed);
    const int64_t max_seq_len = model_args_.max_position_embeddings();
    const uint32_t block_size = options_.block_size();
    const int64_t num_blocks_per_req =
        (max_seq_len + block_size - 1) / block_size + 1;
    auto int_tensor_options = tensor_options_.dtype(torch::kInt32);
    auto token_ids = torch::full({batch_size}, 1, int_tensor_options);
    auto positions = torch::full({batch_size}, 1, int_tensor_options);
    auto new_cache_slots =
        torch::randint(0, 10, {batch_size}, int_tensor_options);
    auto block_table = torch::randint(
        0, 10, {batch_size, num_blocks_per_req}, int_tensor_options);
    auto q_seq_lens =
        torch::randint(0, 10, {batch_size + 1}, int_tensor_options);
    auto kv_seq_lens =
        torch::randint(0, 10, {batch_size + 1}, int_tensor_options);
    auto input_embedding =
        torch::randn({batch_size, model_args_.hidden_size()}, tensor_options_) *
        0.1;
    ModelInputParams input_params;
    input_params.batch_forward_type = BatchForwardType::DECODE;
    input_params.dp_global_token_nums = {1};
    input_params.dp_is_decode = {1};
    input_params.new_cache_slots = new_cache_slots;
    input_params.block_tables = block_table;
    input_params.q_seq_lens = q_seq_lens;
    input_params.kv_seq_lens = kv_seq_lens;
    input_params.input_embedding = input_embedding;

    kv_caches_.resize(batch_size);
    return {token_ids, positions, input_params};
  }

  ModelArgs model_args_;
  torch::TensorOptions tensor_options_;
  runtime::Options options_;
  std::unique_ptr<CausalLM> model_;
  std::vector<KVCache> kv_caches_;
  std::unique_ptr<::xllm::mlu::MluGraphExecutorImpl> impl_;
  std::unique_ptr<BaseExecutorImpl> base_impl_;
};

// Test graph creation and execution with different batch sizes
TEST_F(MluGraphExecutorTest, DifferentBatchSizes) {
  // Test with different batch sizes to ensure graph creation works
  const std::vector<uint32_t> batch_sizes = {1, 3, 13, 21, 65};
  for (auto batch_size : batch_sizes) {
    auto forward_input = prepare_inputs(batch_size, 1);
    auto eager_model_output = base_impl_->run({forward_input.token_ids},
                                              {forward_input.positions},
                                              kv_caches_,
                                              {forward_input.input_params});
    auto eager_output = eager_model_output.hidden_states;

    auto graph_model_output = impl_->run({forward_input.token_ids},
                                         {forward_input.positions},
                                         kv_caches_,
                                         {forward_input.input_params});
    auto graph_output = graph_model_output.hidden_states;

    auto replay_model_output = impl_->run({forward_input.token_ids},
                                          {forward_input.positions},
                                          kv_caches_,
                                          {forward_input.input_params});
    auto replay_output = replay_model_output.hidden_states;

    CHECK_EQ(eager_output.sizes(), graph_output.sizes());
    CHECK_EQ(eager_output.sizes(), replay_output.sizes());
    // Compare outputs - should be identical
    torch_mlu::synchronize();
    EXPECT_TRUE(torch::allclose(eager_output, graph_output, 1e-5, 1e-6));
    EXPECT_TRUE(torch::allclose(eager_output, replay_output, 1e-5, 1e-6));
  }
}

// Test multiple runs to verify consistency across different execution modes
TEST_F(MluGraphExecutorTest, MluGraphExecutorVsBaseExecutorImplMultipleRuns) {
  int32_t batch_size = 5;
  int32_t seed = 42;
  auto forward_input = prepare_inputs(batch_size, seed);
  auto eager_model_output = base_impl_->run({forward_input.token_ids},
                                            {forward_input.positions},
                                            kv_caches_,
                                            {forward_input.input_params});
  auto eager_output = eager_model_output.hidden_states;

  auto graph_model_output = impl_->run({forward_input.token_ids},
                                       {forward_input.positions},
                                       kv_caches_,
                                       {forward_input.input_params});
  auto graph_output = graph_model_output.hidden_states;

  CHECK_EQ(eager_output.sizes(), graph_output.sizes());
  // Compare outputs - should be identical
  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(eager_output, graph_output, 1e-5, 1e-6));

  // Run multiple times and compare results
  const int num_runs = 5;
  auto base_forward_input = prepare_inputs(batch_size + 1, seed);
  auto replay_forward_input = prepare_inputs(batch_size + 1, seed);
  EXPECT_TRUE(torch::allclose(base_forward_input.input_params.input_embedding,
                              replay_forward_input.input_params.input_embedding,
                              1e-5,
                              1e-6));

  for (int i = 0; i < num_runs; ++i) {
    auto base_model_output = base_impl_->run({base_forward_input.token_ids},
                                             {base_forward_input.positions},
                                             kv_caches_,
                                             {base_forward_input.input_params});
    auto base_output = base_model_output.hidden_states;

    auto replay_model_output = impl_->run({replay_forward_input.token_ids},
                                          {replay_forward_input.positions},
                                          kv_caches_,
                                          {replay_forward_input.input_params});
    auto replay_output = replay_model_output.hidden_states;
    base_forward_input.input_params.input_embedding = base_output;
    replay_forward_input.input_params.input_embedding = replay_output;
    CHECK_EQ(base_output.sizes(), replay_output.sizes());
  }

  torch_mlu::synchronize();
  EXPECT_TRUE(torch::allclose(base_forward_input.input_params.input_embedding,
                              replay_forward_input.input_params.input_embedding,
                              1e-5,
                              1e-6));
}

}  // namespace xllm
