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

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <memory>
#include <vector>

#include "acl_graph_executor_impl.h"
#include "base_executor_impl.h"
#include "core/framework/batch/batch.h"
#include "core/framework/block/block.h"
#include "core/framework/block/block_manager_impl.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_loader.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_word_embedding_impl.h"
#include "runtime/options.h"

// Global test environment for ACL graph executor tests
class AclGraphExecutorTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    // Initialize glog
    google::InitGoogleLogging("acl_graph_executor_test");
    google::SetStderrLogging(google::INFO);

    // Add any other global initialization here
    std::cout << "Global test environment setup completed" << std::endl;
    int ret = aclrtSetDevice(0);
    if (ret != 0) {
      LOG(ERROR) << "ACL set device id: 0 failed, ret:" << ret;
    }
    torch_npu::init_npu("npu:0");
  }

  void TearDown() override {
    // Cleanup if needed
    google::ShutdownGoogleLogging();
    torch_npu::finalize_npu();
    aclrtResetDevice(0);
    aclFinalize();
    LOG(INFO) << "AclGraphExecutorTestEnvironment TearDown completed.";
  }
};

// Register the global test environment
::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new AclGraphExecutorTestEnvironment);

namespace xllm {

// Initialize glog for testing - use a function to ensure proper initialization
// order
void InitializeGlog() {
  static bool initialized = false;
  if (!initialized) {
    google::InitGoogleLogging("acl_graph_executor_test");
    google::SetStderrLogging(google::INFO);
    initialized = true;
  }
}

// Simple CausalLM implementation for testing ACL graph executor
// Uses basic operations to verify graph capture and replay functionality
class SimpleCausalLM : public CausalLM {
 public:
  SimpleCausalLM(const ModelArgs& args, const torch::Device& device)
      : args_(args), device_(device) {
    // Initialize a simple linear layer for testing
    linear_ = register_module("linear",
                              torch::nn::Linear(torch::nn::LinearOptions(
                                  args.hidden_size(), args.hidden_size())));

    // Initialize token embedding table
    const int64_t vocab_size = std::max(args.vocab_size(), 1000L);
    token_embedding_table_ = register_parameter(
        "token_embedding",
        torch::randn({vocab_size, args.hidden_size()},
                     torch::dtype(torch::kFloat32).device(device)));

    // Initialize position embedding table
    const int64_t max_pos = args.max_position_embeddings();
    pos_embedding_table_ = register_parameter(
        "pos_embedding",
        torch::randn({max_pos, args.hidden_size()},
                     torch::dtype(torch::kFloat32).device(device)));
    // Initialize block-related tensors for pure device computation
    block_size_ = torch::tensor(4L, torch::dtype(torch::kInt64).device(device));
    scalar_one_ = torch::tensor(1L, torch::dtype(torch::kInt64).device(device));

    // Initialize scalar tensors for computation
    // const tensors
    kv_scale_ =
        torch::tensor(0.01f, torch::dtype(torch::kFloat32).device(device));
    q_scale_ =
        torch::tensor(0.01f, torch::dtype(torch::kFloat32).device(device));
    cache_scale_ =
        torch::tensor(0.005f, torch::dtype(torch::kFloat32).device(device));
    block_scale_ =
        torch::tensor(0.001f, torch::dtype(torch::kFloat32).device(device));

    // Move to device
    this->to(device);
  }

  torch::Tensor forward_impl(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_caches,
                             const ModelInputParams& params) {
    // Simple computation: token embedding + position embedding + linear layer
    // This creates temporary tensors that NPUGraph mempool will manage
    LOG(INFO) << "SimpleCausalLM forward_impl, tokens: " << tokens.sizes()
              << ", positions: " << positions.sizes()
              << ", kv_caches: " << kv_caches.size()
              << ", params: " << params.num_sequences;
    const int64_t num_tokens = tokens.size(0);
    const int64_t hidden_size = args_.hidden_size();

    // Create token embeddings using standard embedding lookup
    auto token_embeddings = torch::embedding(token_embedding_table_, tokens);

    // Create position embeddings using standard embedding lookup
    auto position_embeddings =
        torch::embedding(pos_embedding_table_, positions);

    // Combine embeddings
    auto combined = token_embeddings + position_embeddings;

    // Apply linear layer
    auto output = linear_->forward(combined);

    // Add some computation using other params to make it more realistic
    // if (params.kv_seq_lens.defined()) {
    //   // Use kv_seq_lens in computation
    //   auto kv_lens_sum = torch::sum(params.kv_seq_lens);
    //   output = output + kv_lens_sum * kv_scale_;
    // }

    // if (params.q_seq_lens.defined()) {
    //   // Use q_seq_lens in computation
    //   auto q_lens_sum = torch::sum(params.q_seq_lens);
    //   output = output + q_lens_sum * q_scale_;
    // }

    if (params.new_cache_slots.defined()) {
      // Use new_cache_slots in computation
      auto cache_slots_sum = torch::sum(params.new_cache_slots);
      output = output + cache_slots_sum * cache_scale_;
    }

    if (params.block_tables.defined() && !kv_caches.empty()) {
      // Use block_tables to do embedding lookup from kv_cache - pure device
      // computation Calculate max_seq_len from actual seq_len tensor
      auto max_seq_len = torch::max(params.kv_seq_lens);

      // Calculate max_block_nums_per_seq
      auto max_block_nums_per_seq = torch::ceil(max_seq_len / block_size_);

      // Get kv_cache tensor from KVCache
      const auto& kv_cache_tensor = kv_caches[0].get_k_cache();

      // Create col_indices and mask
      int64_t block_table_len = params.block_tables.size(1);
      auto col_indices = torch::arange(
          block_table_len, torch::dtype(torch::kInt64).device(device_));
      auto mask = col_indices < (max_block_nums_per_seq - scalar_one_);

      // Directly compute embedding
      auto kv_embeddings =
          torch::embedding(kv_cache_tensor, params.block_tables);

      // Apply mask and sum
      auto kv_embeddings_masked = kv_embeddings * mask.view({1, -1, 1});
      auto kv_embeddings_sum = torch::sum(kv_embeddings_masked);
      output = output + kv_embeddings_sum * block_scale_;
    }

    return output;
  }

  // Adapter method to match CausalLM base class interface
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& parameters) override {
    return forward_impl(tokens, positions, kv_caches, parameters);
  }

  const torch::TensorOptions& options() const override {
    static torch::TensorOptions opts =
        torch::dtype(torch::kFloat32).device(device_);
    return opts;
  }

  const ModelArgs& args() const { return args_; }

  // Implement required virtual functions
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override {
    // Simple logits computation
    const int64_t vocab_size = std::max(args_.vocab_size(), 1000L);
    return torch::randn({hidden_states.size(0), vocab_size},
                        torch::dtype(torch::kFloat32).device(device_));
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    // Simple implementation for testing
  }

  torch::Device device() const override { return device_; }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    // Simple implementation for testing
  }

  void update_expert_weight(int32_t layer_id) override {
    // Simple implementation for testing
  }

  layer::NpuLmHead get_npu_lm_head() override {
    // Simple implementation for testing
    return layer::NpuLmHead(nullptr);
  }

  void set_npu_lm_head(layer::NpuLmHead& head) override {
    // Simple implementation for testing
  }

  layer::NpuWordEmbedding get_npu_word_embedding() override {
    // Simple implementation for testing
    return layer::NpuWordEmbedding(nullptr);
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) override {
    // Simple implementation for testing
  }

 private:
  ModelArgs args_;
  torch::Device device_;
  torch::nn::Linear linear_{nullptr};
  torch::Tensor token_embedding_table_;
  torch::Tensor pos_embedding_table_;

  // Pre-allocated constant scalar tensors for computation
  torch::Tensor kv_scale_;
  torch::Tensor q_scale_;
  torch::Tensor cache_scale_;
  torch::Tensor block_scale_;
  torch::Tensor block_size_;
  torch::Tensor scalar_one_;
};

class AclGraphExecutorTest : public ::testing::Test {
 protected:
  AclGraphExecutorTest() = default;

  void SetUp() override {
    if (initialized_) {
      return;
    }
    initialized_ = true;
    sequences_.reserve(100);

    // Set up model args
    model_args_.model_type("test_model");
    model_args_.dtype("float32");
    model_args_.hidden_size(128);
    model_args_.max_position_embeddings(2048);
    model_args_.vocab_size(1000);  // Set a reasonable vocab size

    // Set up device
    device_ = std::make_unique<torch::Device>("npu:0");

    // Set up runtime options
    options_.num_decoding_tokens(1);
    options_.block_size(4);

    // Create simple model
    model_ = std::make_unique<SimpleCausalLM>(model_args_, *device_);

    // Initialize block manager
    const uint32_t n_blocks = 1000;
    const uint32_t block_size = 4;
    BlockManager::Options block_options;
    block_options.num_blocks(n_blocks).block_size(block_size);
    block_manager_ = std::make_unique<BlockManagerImpl>(block_options);

    // Initialize sampling and stopping parameters
    sampling_param_.frequency_penalty = 0.1;
    stopping_checker_.set_max_generated_tokens(20);

    // Initialize sequence parameters
    seq_params_.seq_capacity = 100;
    seq_params_.stopping_checker = &stopping_checker_;
    seq_params_.sampling_param = &sampling_param_;
    seq_params_.skip_special_tokens = true;
    seq_params_.echo = false;
    seq_params_.logprobs = false;
    seq_params_.enable_schedule_overlap = false;

    // Initialize input embedding and mm_data
    input_embedding_ =
        torch::zeros({1, model_args_.hidden_size()},
                     torch::dtype(torch::kFloat32).device(*device_));
    mm_data_ = MMData();  // Default constructor creates empty MMData

    // Initialize KV caches
    kv_caches_.clear();
    const int64_t hidden_size = model_args_.hidden_size();

    // Create KV cache with shape [n_blocks, block_size, hidden_size]
    auto kv_cache =
        torch::randn({n_blocks, block_size * hidden_size},
                     torch::dtype(torch::kFloat32).device(*device_));
    kv_caches_.push_back({kv_cache, kv_cache});
  }

  void TearDown() override { return; }

  void reset() {
    for (auto& sequence : sequences_) {
      auto blocks = sequence.kv_state().kv_blocks();
      if (!blocks.empty()) {
        block_manager_->deallocate(blocks);
      }
    }
  }

  // Helper function to create a simple batch
  std::unique_ptr<Batch> CreateTestBatch() {
    sequences_.emplace_back(0,
                            std::vector<int32_t>{1, 3, 5, 7, 5, 4, 3, 2, 1},
                            input_embedding_,
                            mm_data_,
                            fake_decoder_,
                            seq_params_);
    auto& sequence = sequences_.back();

    // Allocate blocks and configure sequence
    sequence.add_kv_blocks(block_manager_->allocate(3));
    // Set kv_cache_tokens_num to be >= num_prompt_tokens to move to decode
    // stage
    sequence.kv_state().incr_kv_cache_tokens_num(
        /*size=*/9);  // 9 prompt tokens
    sequence.append_token(100);

    // Create batch with pointer to sequence (batch doesn't own sequence)
    auto batch = std::make_unique<Batch>();
    batch->add(&sequence);

    return batch;
  }
  bool initialized_ = false;
  ModelArgs model_args_;
  std::unique_ptr<torch::Device> device_;
  runtime::Options options_;
  std::unique_ptr<CausalLM> model_;

  // Shared resources for all tests
  std::unique_ptr<BlockManagerImpl> block_manager_;
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  SequenceParams seq_params_;
  torch::Tensor input_embedding_;
  MMData mm_data_;
  std::vector<KVCache> kv_caches_;

  // Sequences managed by test class
  std::vector<Sequence> sequences_;

  // Create a sequence in decode phase
  IncrementalDecoder fake_decoder_ = IncrementalDecoder("", 1, false, false);
};

// Test that ACL graph executor produces same results as eager execution
TEST_F(AclGraphExecutorTest, GraphExecutorVsEagerExecution) {
  // Create test batch
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  // Prepare forward input
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);

  std::cout << "forward_input.token_ids: " << forward_input.token_ids
            << std::endl;
  std::cout << "forward_input.positions: " << forward_input.positions
            << std::endl;
  std::cout << "forward_input.input_params.q_seq_lens: "
            << forward_input.input_params.q_seq_lens << std::endl;
  std::cout << "forward_input.input_params.kv_seq_lens: "
            << forward_input.input_params.kv_seq_lens << std::endl;
  std::cout << "forward_input.input_params.new_cache_slots: "
            << forward_input.input_params.new_cache_slots << std::endl;
  std::cout << "forward_input.input_params.block_tables: "
            << forward_input.input_params.block_tables << std::endl;
  // Test eager execution (direct model forward)
  auto eager_output = model_->forward({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_caches_,
                                      {forward_input.input_params});
  // Create ACL graph executor
  auto graph_executor = std::make_unique<AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  // Test graph execution with NPUGraph mempool optimization
  auto graph_output = graph_executor->run({forward_input.token_ids},
                                          {forward_input.positions},
                                          kv_caches_,
                                          {forward_input.input_params});
  // Compare outputs - should be identical
  EXPECT_TRUE(
      torch::allclose(eager_output, graph_output, /*rtol=*/1e-5, /*atol=*/1e-6))
      << "Eager output:\n"
      << eager_output << "\nGraph output:\n"
      << graph_output;
}

// Test that graph replay produces consistent results across multiple runs
TEST_F(AclGraphExecutorTest, GraphReplayConsistency) {
  // Create test batch
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  // Prepare forward input
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);

  // Create ACL graph executor
  auto graph_executor = std::make_unique<AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  // First execution (should create graph with NPUGraph mempool)
  auto output1 = graph_executor->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params});

  // Second execution (should replay graph using mempool-managed tensors)
  auto output2 = graph_executor->run({forward_input.token_ids},
                                     {forward_input.positions},
                                     kv_caches_,
                                     {forward_input.input_params});

  // Compare outputs - should be identical
  EXPECT_TRUE(torch::allclose(output1, output2, /*rtol=*/1e-5, /*atol=*/1e-6))
      << "First output:\n"
      << output1 << "\nSecond output:\n"
      << output2;
}

// Test graph creation and execution with different batch sizes
TEST_F(AclGraphExecutorTest, DifferentBatchSizes) {
  // Test with different batch sizes to ensure graph creation works
  const std::vector<uint32_t> batch_sizes = {1, 2, 4};

  for (auto batch_size : batch_sizes) {
    // Clear sequences from previous iteration to avoid block exhaustion
    sequences_.clear();

    // Create multiple sequences for larger batch sizes
    auto batch = std::make_unique<Batch>();

    for (uint32_t i = 0; i < batch_size; ++i) {
      sequences_.emplace_back(i,
                              std::vector<int32_t>{static_cast<int32_t>(1 + i),
                                                   static_cast<int32_t>(3 + i),
                                                   static_cast<int32_t>(5 + i),
                                                   static_cast<int32_t>(7 + i)},
                              input_embedding_,
                              mm_data_,
                              fake_decoder_,
                              seq_params_);
      auto& sequence = sequences_.back();
      sequence.add_kv_blocks(block_manager_->allocate(2));
      std::cout << "batch_size: " << batch_size << " i: " << i
                << " sequence.kv_state().current_max_tokens_capacity(): "
                << sequence.kv_state().current_max_tokens_capacity()
                << std::endl;
      // Set kv_cache_tokens_num to be >= num_prompt_tokens to move to decode
      // stage
      sequence.kv_state().incr_kv_cache_tokens_num(
          /*size=*/4);  // 4 prompt tokens

      sequence.append_token(100 + i);
      // Add sequence pointer to batch (batch doesn't own sequence)
      batch->add(&sequence);
    }

    // Prepare forward input
    auto forward_input = batch->prepare_forward_input(
        options_.num_decoding_tokens(), 0, model_args_);
    forward_input = forward_input.to(*device_, torch::kFloat32);
    // Create ACL graph executor
    auto graph_executor =
        new AclGraphExecutorImpl(model_.get(), model_args_, *device_, options_);

    // Test graph execution
    auto output = graph_executor->run({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_caches_,
                                      {forward_input.input_params});

    // Verify output shape
    EXPECT_EQ(output.size(0), batch_size * options_.num_decoding_tokens())
        << "Batch size: " << batch_size;
    EXPECT_EQ(output.size(1), model_args_.hidden_size())
        << "Batch size: " << batch_size;
  }
}

// Test ACL graph executor against original NPU executor implementation
TEST_F(AclGraphExecutorTest, AclGraphExecutorVsBaseExecutorImpl) {
  // Create test batch
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  // Prepare forward input
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);
  // Test NPU Executor Impl (original implementation)
  auto npu_executor = std::make_unique<BaseExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  auto npu_output = npu_executor->run({forward_input.token_ids},
                                      {forward_input.positions},
                                      kv_caches_,
                                      {forward_input.input_params});

  // Test ACL Graph Executor with NPUGraph mempool optimization
  auto graph_executor = std::make_unique<AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  auto graph_output = graph_executor->run({forward_input.token_ids},
                                          {forward_input.positions},
                                          kv_caches_,
                                          {forward_input.input_params});

  // Compare outputs - should be identical
  EXPECT_TRUE(
      torch::allclose(npu_output, graph_output, /*rtol=*/1e-5, /*atol=*/1e-6))
      << "NPU Executor output:\n"
      << npu_output << "\nACL Graph Executor output:\n"
      << graph_output;

  // Verify output shapes are the same
  EXPECT_EQ(npu_output.sizes(), graph_output.sizes())
      << "Output shape mismatch: NPU=" << npu_output.sizes()
      << ", Graph=" << graph_output.sizes();
}

// Test multiple runs to verify consistency across different execution modes
TEST_F(AclGraphExecutorTest, AclGraphExecutorVsBaseExecutorImplMultipleRuns) {
  // Create test batch
  auto batch = CreateTestBatch();
  ASSERT_FALSE(batch->empty());

  // Prepare forward input
  auto forward_input = batch->prepare_forward_input(
      options_.num_decoding_tokens(), 0, model_args_);
  forward_input = forward_input.to(*device_, torch::kFloat32);
  // Create both executors
  auto npu_executor = std::make_unique<BaseExecutorImpl>(
      model_.get(), model_args_, *device_, options_);
  auto graph_executor = std::make_unique<AclGraphExecutorImpl>(
      model_.get(), model_args_, *device_, options_);

  // Run multiple times and compare results
  const int num_runs = 3;
  for (int i = 0; i < num_runs; ++i) {
    // Direct model forward call (baseline)
    auto direct_output = model_->forward({forward_input.token_ids},
                                         {forward_input.positions},
                                         kv_caches_,
                                         {forward_input.input_params});

    // NPU Executor run
    auto npu_output = npu_executor->run({forward_input.token_ids},
                                        {forward_input.positions},
                                        kv_caches_,
                                        {forward_input.input_params});

    // ACL Graph Executor run with NPUGraph mempool
    auto graph_output = graph_executor->run({forward_input.token_ids},
                                            {forward_input.positions},
                                            kv_caches_,
                                            {forward_input.input_params});

    // Compare direct model output with NPU Executor output
    EXPECT_TRUE(torch::allclose(
        direct_output, npu_output, /*rtol=*/1e-5, /*atol=*/1e-6))
        << "Run " << i << " - Direct model vs NPU Executor mismatch:\n"
        << "Direct model output:\n"
        << direct_output << "\nNPU Executor output:\n"
        << npu_output;

    // Compare direct model output with ACL Graph Executor output
    EXPECT_TRUE(torch::allclose(
        direct_output, graph_output, /*rtol=*/1e-5, /*atol=*/1e-6))
        << "Run " << i << " - Direct model vs ACL Graph Executor mismatch:\n"
        << "Direct model output:\n"
        << direct_output << "\nACL Graph Executor output:\n"
        << graph_output;

    // Compare NPU Executor output with ACL Graph Executor output
    EXPECT_TRUE(
        torch::allclose(npu_output, graph_output, /*rtol=*/1e-5, /*atol=*/1e-6))
        << "Run " << i << " - NPU Executor vs ACL Graph Executor mismatch:\n"
        << "NPU Executor output:\n"
        << npu_output << "\nACL Graph Executor output:\n"
        << graph_output;
  }
}

}  // namespace xllm
