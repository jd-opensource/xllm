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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <sstream>

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_MLU)
#include "../../mlu/attention.h"
#elif defined(USE_CUDA)
#include "../../cuda/attention.h"
#endif
#include "layers/common/indexer.h"
#include "platform/device.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {

class MockDeepseekScalingRotaryEmbedding
    : public DeepseekScalingRotaryEmbedding {
 public:
  MockDeepseekScalingRotaryEmbedding(int64_t rotary_dim,
                                     int64_t max_position_embeddings,
                                     int64_t rope_theta,
                                     bool interleaved,
                                     const torch::TensorOptions& options)
      : DeepseekScalingRotaryEmbedding(rotary_dim,
                                       rotary_dim,
                                       max_position_embeddings,
                                       max_position_embeddings,
                                       rope_theta,
                                       interleaved,
                                       /*scaling_factor=*/2.5,
                                       /*extrapolation_factor=*/1.,
                                       /*attn_factor=*/40,
                                       /*beta_fast=*/32,
                                       /*beta_slow=*/1,
                                       /*mscale=*/1.,
                                       /*mscale_all_dim=*/1.,
                                       options) {
    mock_rope_ = RotaryEmbedding(
        rotary_dim, max_position_embeddings, rope_theta, interleaved, options);
  }
  void forward(torch::Tensor& q,
               torch::Tensor& k,
               const torch::Tensor& positions,
               const torch::Tensor& cu_query_lens,
               int64_t max_query_len,
               bool is_prompt) {
    return mock_rope_(q, k, positions, cu_query_lens, max_query_len, is_prompt);
  }

 private:
  RotaryEmbedding mock_rope_{nullptr};
};

class IndexerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize default model arguments for testing
    model_args_ = test::create_default_model_args();

    // Initialize w8a8 quantization arguments
    quant_args_ = test::create_default_quant_args();

    // Initialize tensor options
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Device::type_torch(), 0)
                   .requires_grad(false);

    // Create mock ProcessGroup and initialize ParallelArgs
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);

    // Note: Indexer will be created by individual test cases with their desired
    // dimensions
  }

  void TearDown() override {
    // Clean up if needed
  }

  // Helper function to create test weights for the Indexer (w8a8 smoothquant
  // format)
  std::unordered_map<std::string, torch::Tensor> create_test_weights(
      int64_t dim,
      int64_t index_n_heads,
      int64_t index_head_dim,
      int64_t q_lora_rank) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    // Create weights for wq_b (query projection with LoRA)
    // Shape: [n_heads * head_dim, q_lora_rank]
    auto wq_b_weight = torch::full(
        {index_n_heads * index_head_dim, q_lora_rank}, 0.1f, options_);

    // Create weights for wk (key projection)
    // Shape: [head_dim, dim]
    auto wk_weight = torch::full({index_head_dim, dim}, 0.1f, options_);

    // Create weights for weights_proj (weights projection)
    // Shape: [n_heads, dim]
    auto weights_proj_weight =
        torch::full({index_n_heads, dim}, 0.1f, options_);

    // Create StateDict with w8a8 smoothquant weights
    weight_dict["wq_b.weight"] = wq_b_weight;
    weight_dict["wk.weight"] = wk_weight;
    weight_dict["weights_proj.weight"] = weights_proj_weight;

    LOG(INFO) << "Test bfloat16 weights created successfully";
    LOG(INFO) << "wq_b weight shape: " << weight_dict["wq_b.weight"].sizes();
    LOG(INFO) << "wk weight shape: " << weight_dict["wk.weight"].sizes();
    LOG(INFO) << "weights_proj weight shape: "
              << weight_dict["weights_proj.weight"].sizes();

    return weight_dict;
  }

  // Helper function to populate AttentionMetadata for testing
  void populate_attention_metadata(AttentionMetadata& metadata,
                                   int64_t batch_size,
                                   int64_t max_query_len,
                                   int64_t max_seq_len,
                                   bool is_prefill,
                                   int64_t max_num_batched_tokens) {
    // Create q_cu_seq_lens tensor (cu_seq_q_lens)
    // shape = [batch_size + 1], typically [0, 4, 8, 12, ...] if max_query_len=4
    metadata.q_cu_seq_lens = torch::arange(
        0,
        (batch_size + 1) * max_query_len,
        max_query_len,
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

    // Create kv_cu_seq_lens tensor
    // TODO: Define proper shape and values based on actual requirements
    metadata.kv_cu_seq_lens = torch::zeros(
        {batch_size + 1},
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

    // Create seq_lens tensor
    // Shape: [batch_size]
    metadata.kv_seq_lens = torch::full(
        {batch_size},
        max_query_len,
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

    // Create block_table tensor directly assigned to metadata
    metadata.block_table = torch::zeros(
        {batch_size, max_num_batched_tokens},
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

    // Fill each batch with consecutive numbers
    for (int64_t b = 0; b < batch_size; ++b) {
      int64_t start_val = b * max_query_len + 1;
      int64_t end_val = start_val + max_query_len;
      // Generate sequence [start_val, ..., end_val-1]
      auto seq = torch::arange(start_val,
                               end_val,
                               torch::TensorOptions()
                                   .dtype(torch::kInt32)
                                   .device(options_.device()));
      metadata.block_table[b].index_put_(
          {torch::indexing::Slice(0, max_query_len)}, seq);
    }

    // Create slot_mapping tensor directly assigned to metadata
    metadata.slot_mapping = torch::arange(
        1,
        batch_size * max_query_len + 1,
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

    metadata.max_query_len = max_query_len;
    metadata.max_seq_len = max_seq_len;
    metadata.compute_dtype = "bfloat16";
    metadata.is_prefill = is_prefill;
    metadata.is_chunked_prefill = false;
  }

  // Helper function to create k_cache tensor
  torch::Tensor create_k_cache(int64_t block_num,
                               int64_t block_size,
                               int64_t head_kv,
                               int64_t head_dim,
                               float value = 0.5f) {
    return torch::full(
        {block_num, head_kv, block_size, head_dim}, value, options_);
  }

  // Helper function to verify tensor values are close to expected
  void verify_tensor_close(const torch::Tensor& actual,
                           const torch::Tensor& expected,
                           double rtol = 1e-5,
                           double atol = 1e-8) {
    test::verify_tensor_close(actual, expected, rtol, atol);
  }

  // Helper function to create custom input tensor for precision testing
  torch::Tensor create_custom_input(const std::vector<int64_t>& shape,
                                    const std::vector<float>& values) {
    return test::create_custom_input(shape, values, options_);
  }

  // Helper function to set expected output for precision verification
  void set_expected_output(const std::vector<float>& expected_values) {
    expected_output_ = expected_values;
  }

  // Helper function to verify precision against expected output
  void verify_precision(const torch::Tensor& actual_output,
                        double rtol = 1e-3,
                        double atol = 1e-4) {
    test::verify_precision(actual_output, expected_output_, rtol, atol);
  }

  // Helper function to run Indexer test with configurable batch size, query
  // length and prefill mode
  std::tuple<torch::Tensor, torch::Tensor>
  run_indexer_test(int64_t batch_size, int64_t max_query_len, bool is_prefill) {
    // Fixed configuration parameters
    const int64_t dim = 7168;
    const int64_t index_n_heads = 64;
    const int64_t index_head_dim = 128;
    const int64_t qk_rope_head_dim = 64;
    const int64_t index_topk = 2048;
    const int64_t q_lora_rank = 1536;
    const int64_t max_position_embeddings = 8192;
    const int64_t rope_theta = 10000.0;
    const bool rope_interleaved = true;

    // Config for k cache
    const int64_t head_kv = 1;
    const int64_t block_size = 1;
    const int64_t block_num = 10240;

    int64_t num_tokens = batch_size * max_query_len;

    // Create non-quantized quant_args for bfloat16 mode
    QuantArgs bfloat16_quant_args;  // Empty means no quantization

    std::unique_ptr<DeepseekScalingRotaryEmbedding> rotary_emb =
        std::make_unique<MockDeepseekScalingRotaryEmbedding>(
            qk_rope_head_dim,
            max_position_embeddings,
            rope_theta,
            rope_interleaved,
            options_);
    auto indexer = Indexer(IndexerImpl(dim,
                                       index_n_heads,
                                       index_head_dim,
                                       qk_rope_head_dim,
                                       index_topk,
                                       q_lora_rank,
                                       *rotary_emb,
                                       bfloat16_quant_args,
                                       parallel_args_,
                                       options_));

    // Create test weights
    std::unordered_map<std::string, torch::Tensor> weight_dict;
    auto wq_b_weight = torch::full(
        {index_n_heads * index_head_dim, q_lora_rank}, 0.1f, options_);
    auto wk_weight = torch::full({index_head_dim, dim}, 0.1f, options_);
    auto weights_proj_weight =
        torch::full({index_n_heads, dim}, 0.1f, options_);

    weight_dict["wq_b.weight"] = wq_b_weight;
    weight_dict["wk.weight"] = wk_weight;
    weight_dict["weights_proj.weight"] = weights_proj_weight;

    StateDict state_dict(weight_dict);
    indexer->load_state_dict(state_dict);

    // Create test inputs
    auto x = torch::ones({num_tokens, dim}, options_);
    auto qr = torch::ones({num_tokens, q_lora_rank}, options_);
    // Generate positions: [0, 1, ..., max_query_len-1] repeated batch_size
    // times
    auto positions = torch::arange(max_query_len,
                                   torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(options_.device()))
                         .repeat({batch_size});
    auto k_cache =
        torch::zeros({block_num, head_kv, block_size, index_head_dim},
                     torch::TensorOptions()
                         .dtype(torch::kBFloat16)
                         .device(options_.device()));

    // Create metadata object and populate it
    AttentionMetadata metadata;
    populate_attention_metadata(metadata,
                                batch_size,
                                max_query_len,
                                max_position_embeddings,
                                is_prefill,
                                num_tokens);

    // Test forward pass and return results
    return indexer->forward(x, qr, positions, k_cache, metadata, is_prefill);
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;

  // Helper to create a mock ProcessGroup for testing
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;

  // Expected output for precision verification
  std::vector<float> expected_output_;
};

TEST_F(IndexerTest, Bfloat16PrefillVerifyPrecision) {
  // Test bfloat16 mode (non-quantized) - prefill phase
  // 4K test
  int64_t batch_size = 2;
  int64_t max_query_len = 4096;
  const bool is_prefill = true;
  const int64_t index_topk = 2048;

  int64_t num_tokens = batch_size * max_query_len;

  // Run the test using the encapsulated function
  auto [new_block_tables, new_context_lens] =
      run_indexer_test(batch_size, max_query_len, is_prefill);

  // Verify output shapes
  ASSERT_EQ(new_block_tables.sizes().size(), 2)
      << "new_block_tables should be 2D tensor";
  ASSERT_EQ(new_context_lens.sizes().size(), 1)
      << "new_context_lens should be 1D tensor";
  ASSERT_EQ(new_block_tables.size(0), num_tokens) << "Batch size should match";
  ASSERT_EQ(new_block_tables.size(1), index_topk) << "Top-k should match";

  // Verify that the first value in new_block_tables is 1 (calculated via vLLM
  // MLU)
  ASSERT_EQ(new_block_tables.index({0, 0}).item<int64_t>(), 1)
      << "The first value in new_block_tables should be 1";

  // Test bfloat16 mode (non-quantized) - prefill phase
  // 8K test
  max_query_len = 8192;

  num_tokens = batch_size * max_query_len;

  // Run the test using the encapsulated function
  std::tie(new_block_tables, new_context_lens) =
      run_indexer_test(batch_size, max_query_len, is_prefill);

  // Verify output shapes
  ASSERT_EQ(new_block_tables.sizes().size(), 2)
      << "new_block_tables should be 2D tensor";
  ASSERT_EQ(new_context_lens.sizes().size(), 1)
      << "new_context_lens should be 1D tensor";
  ASSERT_EQ(new_block_tables.size(0), num_tokens) << "Batch size should match";
  ASSERT_EQ(new_block_tables.size(1), index_topk) << "Top-k should match";

  // Verify that the first value in new_block_tables is 1 (calculated via vLLM
  // MLU)
  ASSERT_EQ(new_block_tables.index({0, 0}).item<int64_t>(), 1)
      << "The first value in new_block_tables should be 1";
}

TEST_F(IndexerTest, Bfloat16DecodeVerifyPrecision) {
  // Test bfloat16 mode (non-quantized) - decode phase
  const int64_t batch_size = 2048;
  const int64_t max_query_len = 1;
  const bool is_prefill = false;

  int64_t num_tokens = batch_size * max_query_len;

  // Run the test using the encapsulated function
  auto [new_block_tables, new_context_lens] =
      run_indexer_test(batch_size, max_query_len, is_prefill);

  // Verify output shapes
  ASSERT_EQ(new_block_tables.sizes().size(), 2)
      << "new_block_tables should be 2D tensor";
  ASSERT_EQ(new_context_lens.sizes().size(), 1)
      << "new_context_lens should be 1D tensor";
  ASSERT_EQ(new_block_tables.size(0), num_tokens) << "Batch size should match";
  ASSERT_EQ(new_block_tables.size(1), 2048) << "Top-k should match";

  // Verify that the first value in new_block_tables is 1 (calculated via vLLM
  // MLU)
  ASSERT_EQ(new_block_tables.index({0, 0}).item<int64_t>(), 1)
      << "The first value in new_block_tables should be 1";
  // Verify that all values in new_context_lens are 1
  for (int64_t i = 0; i < new_context_lens.size(0); ++i) {
    ASSERT_EQ(new_context_lens.index({i}).item<int64_t>(), 1)
        << "All values in new_context_lens should be 1";
  }
}

}  // namespace layer
}  // namespace xllm
