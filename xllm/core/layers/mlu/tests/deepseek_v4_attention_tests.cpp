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

#include <array>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/tests/tests_utils.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_attention.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

namespace test {

void verify_sum_min_max(const torch::Tensor& output,
                        float expected_sum,
                        float expected_min,
                        float expected_max,
                        float rtol = 1e-2) {
  torch::Tensor output_fp32 = output.to(torch::kFloat32).flatten();
  float actual_sum = torch::sum(output_fp32).item<float>();
  float actual_min = torch::min(output_fp32).item<float>();
  float actual_max = torch::max(output_fp32).item<float>();

  LOG(INFO) << "  sum: " << actual_sum << " (expected: " << expected_sum << ")";
  LOG(INFO) << "  min: " << actual_min << " (expected: " << expected_min << ")";
  LOG(INFO) << "  max: " << actual_max << " (expected: " << expected_max << ")";

  EXPECT_NEAR(actual_sum, expected_sum, std::abs(expected_sum) * rtol + 1.0f)
      << "Sum mismatch";
  EXPECT_NEAR(actual_min, expected_min, std::abs(expected_min) * rtol + 0.1f)
      << "Min mismatch";
  EXPECT_NEAR(actual_max, expected_max, std::abs(expected_max) * rtol + 0.1f)
      << "Max mismatch";
}

}  // namespace test

// Test configuration parameters matching Python test
struct DeepSeekV4AttentionTestConfig {
  int64_t dim = 4096;
  int64_t head_dim = 512;
  int64_t n_heads = 64;
  int64_t q_lora_rank = 1024;
  int64_t rope_head_dim = 64;
  int64_t o_groups = 8;
  int64_t o_lora_rank = 1024;
  int64_t window_size = 128;
  int64_t compress_ratio = 4;
  int64_t max_position_embeddings = 65536;
  int64_t original_seq_len = 65536;
  double rope_theta = 10000.0;
  double compress_rope_theta = 160000.0;
  double norm_eps = 1e-6;
  int64_t index_n_heads = 64;
  int64_t index_head_dim = 128;
  int64_t index_topk = 512;
  int64_t cached_state_num = 4;

  // Test input parameters
  int64_t batch_size = 2;
  int64_t seq_len = 1024;

  // Weight initialization values (matching Python test)
  float weight_val = 0.1f;
  float bias_val = 0.2f;
  float input_val = 0.5f;
};

class DeepSeekV4AttentionTest : public ::testing::Test {
 protected:
  torch::TensorOptions options_;
  torch::TensorOptions options_fp32_;
  torch::TensorOptions int_options_;
  torch::TensorOptions int64_options_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  ParallelArgs parallel_args_{0, 1, nullptr};

  void SetUp() override {
    auto create_options = [this](torch::ScalarType dtype) {
      return torch::TensorOptions()
          .dtype(dtype)
          .device(Device::type_torch(), 0)
          .requires_grad(false);
    };

    options_ = create_options(torch::kBFloat16);
    options_fp32_ = create_options(torch::kFloat32);
    int_options_ = create_options(torch::kInt32);
    int64_options_ = create_options(torch::kInt64);

    mock_process_group_ = std::make_unique<test::MockProcessGroup>(
        torch::Device(Device::type_torch(), 0));
    parallel_args_ = ParallelArgs(0, 1, mock_process_group_.get());
    parallel_args_.tp_group_ = mock_process_group_.get();
    parallel_args_.single_rank_group_ = mock_process_group_.get();
    parallel_args_.sp_group_ = mock_process_group_.get();
  }

  // Calculate num_blocks following Python test logic
  int64_t calculate_num_blocks(const DeepSeekV4AttentionTestConfig& config) {
    int64_t max_blocks_attention =
        config.window_size +
        (config.compress_ratio > 0
             ? config.max_position_embeddings / config.compress_ratio
             : 0);
    int64_t max_blocks_indexer =
        config.compress_ratio > 0 ? config.seq_len / config.compress_ratio + 100
                                  : 0;
    return std::max(max_blocks_attention, max_blocks_indexer);
  }

  // Create ModelArgs from test config
  ModelArgs create_model_args(const DeepSeekV4AttentionTestConfig& config,
                              int64_t layer_id) {
    ModelArgs args;
    args.model_type() = "deepseek_v4";
    args.hidden_size() = config.dim;
    args.head_dim() = config.head_dim;
    args.n_heads() = config.n_heads;
    args.q_lora_rank() = config.q_lora_rank;
    args.rotary_dim() = config.rope_head_dim;
    args.n_group() = config.o_groups;
    args.kv_lora_rank() = config.o_lora_rank;
    args.sliding_window() = config.window_size;
    args.max_position_embeddings() = config.max_position_embeddings;
    args.rope_scaling_original_max_position_embeddings() =
        config.original_seq_len;
    args.rope_theta() = static_cast<float>(config.rope_theta);
    args.rope_scaling_high_freq_factor() =
        static_cast<float>(config.compress_rope_theta);
    args.rope_scaling_factor() = 1.0f;
    args.rope_extrapolation_factor() = 1.0f;
    args.rope_scaling_attn_factor() = 1.0f;
    args.rope_scaling_beta_fast() = 32;
    args.rope_scaling_beta_slow() = 1;
    args.rope_scaling_mscale() = 1.0f;
    args.rope_scaling_mscale_all_dim() = 1;

    args.rms_norm_eps() = static_cast<float>(config.norm_eps);
    args.index_n_heads() = config.index_n_heads;
    args.index_head_dim() = config.index_head_dim;
    args.index_topk() = config.index_topk;

    // Create compress_ratios array with config.compress_ratio for all layers
    args.rope_scaling() = config.compress_ratio;

    return args;
  }

  // Create weight dictionary for DeepSeekV4Attention
  std::unordered_map<std::string, torch::Tensor> create_attention_weights(
      const DeepSeekV4AttentionTestConfig& config) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    const int64_t num_local_heads = config.n_heads;
    const int64_t o_local_groups = config.o_groups;

    // Query projection weights (LoRA)
    if (config.q_lora_rank > 0) {
      weight_dict["wq_a.weight"] = torch::full(
          {config.q_lora_rank, config.dim}, config.weight_val, options_);
      weight_dict["q_norm.weight"] =
          torch::full({config.q_lora_rank}, config.weight_val, options_);
      weight_dict["wq_b.weight"] =
          torch::full({num_local_heads * config.head_dim, config.q_lora_rank},
                      config.weight_val,
                      options_);
    }

    // KV projection weights
    weight_dict["wkv.weight"] =
        torch::full({config.head_dim, config.dim}, config.weight_val, options_);
    weight_dict["kv_norm.weight"] =
        torch::full({config.head_dim}, config.weight_val, options_);

    // Output projection weights
    weight_dict["wo_a.weight"] =
        torch::full({o_local_groups * config.o_lora_rank,
                     num_local_heads * config.head_dim / config.o_groups},
                    config.weight_val,
                    options_);
    weight_dict["wo_b.weight"] =
        torch::full({config.dim, o_local_groups * config.o_lora_rank},
                    config.weight_val,
                    options_);

    // Attention sink
    weight_dict["attn_sink"] =
        torch::full({num_local_heads}, config.bias_val, options_fp32_);

    // Compressor weights (only if compress_ratio > 0)
    if (config.compress_ratio > 0) {
      int64_t coeff = (config.compress_ratio == 4) ? 2 : 1;
      int64_t kv_dim = config.head_dim * coeff;

      weight_dict["compressor.ape"] = torch::full(
          {config.compress_ratio, kv_dim}, config.weight_val, options_fp32_);
      weight_dict["compressor.wkv.weight"] =
          torch::full({kv_dim, config.dim}, config.weight_val, options_fp32_);
      weight_dict["compressor.wgate.weight"] =
          torch::full({kv_dim, config.dim}, config.weight_val, options_fp32_);
      weight_dict["compressor.norm.weight"] =
          torch::full({config.head_dim}, config.weight_val, options_);

      // Indexer weights (only if compress_ratio == 4)
      if (config.compress_ratio == 4) {
        weight_dict["indexer.wq_b.weight"] = torch::full(
            {config.index_n_heads * config.index_head_dim, config.q_lora_rank},
            config.weight_val,
            options_);
        weight_dict["indexer.weights_proj.weight"] = torch::full(
            {config.index_n_heads, config.dim}, config.weight_val, options_);

        int64_t indexer_kv_dim = config.index_head_dim * coeff;

        weight_dict["indexer.compressor.ape"] =
            torch::full({config.compress_ratio, indexer_kv_dim},
                        config.weight_val,
                        options_fp32_);
        weight_dict["indexer.compressor.wkv.weight"] = torch::full(
            {indexer_kv_dim, config.dim}, config.weight_val, options_fp32_);
        weight_dict["indexer.compressor.wgate.weight"] = torch::full(
            {indexer_kv_dim, config.dim}, config.weight_val, options_fp32_);
        weight_dict["indexer.compressor.norm.weight"] =
            torch::full({config.index_head_dim}, config.weight_val, options_);
      }
    }

    LOG(INFO) << "Created attention weights with " << weight_dict.size()
              << " tensors";
    return weight_dict;
  }

  // Create AttentionMetadata for testing
  AttentionMetadata create_attention_metadata(
      const DeepSeekV4AttentionTestConfig& config,
      bool is_prefill) {
    AttentionMetadata metadata;
    int64_t batch_size = config.batch_size;
    int64_t seq_len = config.seq_len;
    int64_t num_blocks = calculate_num_blocks(config);

    // q_cu_seq_lens: cumulative sequence lengths [0, seq_len, 2*seq_len, ...]
    std::vector<int32_t> cu_seq_lens_vec(batch_size + 1);
    for (int32_t i = 0; i <= batch_size; ++i) {
      cu_seq_lens_vec[i] = i * seq_len;
    }
    metadata.q_cu_seq_lens = torch::tensor(cu_seq_lens_vec, int_options_);

    // kv_seq_lens and q_seq_lens
    metadata.kv_seq_lens = torch::full({batch_size}, seq_len, int_options_);
    metadata.q_seq_lens = torch::full({batch_size}, seq_len, int_options_);

    // block_table: [batch_size, num_blocks], values: arange(num_blocks)
    metadata.block_table = torch::arange(num_blocks, int_options_)
                               .unsqueeze(0)
                               .repeat({batch_size, 1});

    metadata.max_query_len = seq_len;
    metadata.max_seq_len = seq_len;
    metadata.is_prefill = is_prefill;
    metadata.is_chunked_prefill = false;
    metadata.compute_dtype = "float";

    return metadata;
  }

  // Create KVCache for testing
  KVCache create_kv_cache(const DeepSeekV4AttentionTestConfig& config) {
    int64_t num_blocks = calculate_num_blocks(config);

    // KV cache shape: [num_blocks, 1, 1, head_dim]
    auto key_cache =
        torch::zeros({num_blocks, 1, 1, config.head_dim}, options_);
    auto value_cache = torch::Tensor();

    // Index cache for indexer (if compress_ratio == 4)
    torch::Tensor index_cache;
    if (config.compress_ratio == 4) {
      index_cache =
          torch::zeros({num_blocks, 1, 1, config.index_head_dim}, options_);
    } else {
      index_cache = torch::zeros({0}, options_);
    }

    return KVCache(
        IndexedKVCacheTensors{{key_cache, value_cache}, index_cache});
  }

  void synchronize_device() {
    xllm::Device device(options_.device());
    device.synchronize_default_stream();
  }

  // Verify multiple KV cache blocks
  void verify_kv_cache_blocks(
      const torch::Tensor& cache,
      const std::string& cache_name,
      const std::array<std::array<float, 3>, 3>& expected_values) {
    LOG(INFO) << "Verifying " << cache_name << " (first 3 blocks):";
    LOG(INFO) << cache_name << " shape: " << cache.sizes();

    for (int64_t block_idx = 0; block_idx < 3; ++block_idx) {
      LOG(INFO) << "Block " << block_idx << ":";
      test::verify_sum_min_max(cache.index({block_idx}).flatten(),
                               /*sum=*/expected_values[block_idx][0],
                               /*min=*/expected_values[block_idx][1],
                               /*max=*/expected_values[block_idx][2]);
    }
  }

  // Print tensor info for debugging
  void print_tensor_info(const std::string& name, const torch::Tensor& tensor) {
    auto flat = tensor.flatten().to(torch::kFloat32);
    LOG(INFO) << name << ":";
    LOG(INFO) << "  shape: " << tensor.sizes();
    LOG(INFO) << "  dtype: " << tensor.dtype();
    LOG(INFO) << "  sum: " << torch::sum(flat).item<float>();
    LOG(INFO) << "  min: " << torch::min(flat).item<float>();
    LOG(INFO) << "  max: " << torch::max(flat).item<float>();
  }

  // Create attention module from config
  DeepSeekV4Attention create_attention_module(
      const DeepSeekV4AttentionTestConfig& config,
      int64_t layer_id) {
    auto model_args = create_model_args(config, layer_id);
    QuantArgs quant_args;

    DeepSeekV4Attention attn(DeepSeekV4AttentionImpl(model_args,
                                                     quant_args,
                                                     parallel_args_,
                                                     options_,
                                                     layer_id,
                                                     config.cached_state_num));

    auto weight_dict = create_attention_weights(config);
    torch::NoGradGuard no_grad;
    attn->load_state_dict(StateDict(weight_dict));

    return attn;
  }

  // Create batch_to_kv_state vector
  std::vector<int64_t> create_batch_to_kv_state(int64_t batch_size) {
    std::vector<int64_t> batch_to_kv_state(batch_size);
    std::iota(batch_to_kv_state.begin(), batch_to_kv_state.end(), 0L);
    return batch_to_kv_state;
  }

  // Create prefill positions tensor
  torch::Tensor create_prefill_positions(int64_t seq_len, int64_t batch_size) {
    return torch::arange(seq_len, int_options_).repeat({batch_size});
  }
};

// Test forward pass with compress_ratio = 4 (matches Python test)
TEST_F(DeepSeekV4AttentionTest, ForwardPrefillCompressRatio4Test) {
  DeepSeekV4AttentionTestConfig config;
  config.compress_ratio = 4;

  const int64_t layer_id = 0;
  const int64_t batch_size = config.batch_size;
  const int64_t seq_len = config.seq_len;
  const int64_t num_tokens = batch_size * seq_len;

  LOG(INFO) << "Testing DeepSeekV4Attention forward pass (prefill)";
  LOG(INFO) << "  compress_ratio: " << config.compress_ratio;
  LOG(INFO) << "  batch_size: " << batch_size << ", seq_len: " << seq_len;

  // Create attention module
  auto attn = create_attention_module(config, layer_id);

  // Prepare input data
  auto hidden_states =
      torch::full({num_tokens, config.dim}, config.input_val, options_);
  print_tensor_info("hidden_states", hidden_states);

  // positions: [0, 1, 2, ..., seq_len-1] repeated for batch
  auto positions = create_prefill_positions(seq_len, batch_size);
  LOG(INFO) << "positions shape: " << positions.sizes();

  auto batch_to_kv_state = create_batch_to_kv_state(batch_size);

  // Create attention metadata and KV cache
  auto attn_metadata = create_attention_metadata(config, /*is_prefill=*/true);
  auto kv_cache = create_kv_cache(config);

  // Run forward pass
  LOG(INFO) << "Running forward pass...";
  auto output = attn->forward(
      positions, hidden_states, attn_metadata, kv_cache, batch_to_kv_state);

  synchronize_device();

  // Verify output shape
  LOG(INFO) << "Verifying output...";
  print_tensor_info("output", output);
  ASSERT_EQ(output.dim(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), num_tokens) << "Output batch*seq should match";
  ASSERT_EQ(output.size(1), config.dim)
      << "Output dim should match hidden_size";

  // Verify output values (from Python test log)
  // Expected: sum=267126833152.0, min=29952.0, max=32640.0
  LOG(INFO) << "Verifying output values:";
  test::verify_sum_min_max(output, 2.671268e11f, 29952.0f, 32640.0f);

  // Verify KV cache updates (first 3 blocks)
  auto k_cache = kv_cache.get_k_cache();
  verify_kv_cache_blocks(k_cache,
                         "k_cache",
                         {{/*Block 0:*/ {47.25f, -0.141602f, 0.141602f},
                           /*Block 1:*/ {47.25f, -0.141602f, 0.141602f},
                           /*Block 2:*/ {47.25f, -0.141602f, 0.141602f}}});

  // Verify Indexer KV cache updates (first 3 blocks)
  auto index_cache = kv_cache.get_index_cache();
  verify_kv_cache_blocks(index_cache,
                         "index_cache",
                         {{/*Block 0:*/ {1.132812f, 0.0f, 1.132812f},
                           /*Block 1:*/ {1.132812f, -0.112793f, 1.023438f},
                           /*Block 2:*/ {1.132812f, -0.134766f, 1.0f}}});

  LOG(INFO) << "DeepSeekV4Attention forward test passed!";
}

TEST_F(DeepSeekV4AttentionTest, ForwardPrefillNoCompressTest) {
  DeepSeekV4AttentionTestConfig config;
  config.compress_ratio = 0;
  config.seq_len = 64;

  const int64_t layer_id = 0;
  const int64_t batch_size = config.batch_size;
  const int64_t seq_len = config.seq_len;
  const int64_t num_tokens = batch_size * seq_len;

  auto attn = create_attention_module(config, layer_id);
  auto hidden_states =
      torch::full({num_tokens, config.dim}, config.input_val, options_);
  auto positions = create_prefill_positions(seq_len, batch_size);
  auto batch_to_kv_state = create_batch_to_kv_state(batch_size);
  auto attn_metadata = create_attention_metadata(config, /*is_prefill=*/true);
  auto kv_cache = create_kv_cache(config);

  auto output = attn->forward(
      positions, hidden_states, attn_metadata, kv_cache, batch_to_kv_state);

  synchronize_device();

  ASSERT_EQ(output.dim(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), num_tokens) << "Output batch*seq should match";
  ASSERT_EQ(output.size(1), config.dim)
      << "Output dim should match hidden_size";
  ASSERT_TRUE(torch::isfinite(output.to(torch::kFloat32)).all().item<bool>())
      << "Output should be finite";
}

// Test forward pass with compress_ratio = 128 (matches Python test)
TEST_F(DeepSeekV4AttentionTest, ForwardPrefillCompressRatio128Test) {
  DeepSeekV4AttentionTestConfig config;
  config.compress_ratio = 128;

  const int64_t layer_id = 0;
  const int64_t batch_size = config.batch_size;
  const int64_t seq_len = config.seq_len;
  const int64_t num_tokens = batch_size * seq_len;

  LOG(INFO) << "Testing DeepSeekV4Attention forward pass (prefill)";
  LOG(INFO) << "  compress_ratio: " << config.compress_ratio;
  LOG(INFO) << "  batch_size: " << batch_size << ", seq_len: " << seq_len;

  // Create attention module
  auto attn = create_attention_module(config, layer_id);

  // Prepare input data
  auto hidden_states =
      torch::full({num_tokens, config.dim}, config.input_val, options_);
  print_tensor_info("hidden_states", hidden_states);

  // positions: [0, 1, 2, ..., seq_len-1] repeated for batch
  auto positions = create_prefill_positions(seq_len, batch_size);
  LOG(INFO) << "positions shape: " << positions.sizes();

  auto batch_to_kv_state = create_batch_to_kv_state(batch_size);

  // Create attention metadata and KV cache
  auto attn_metadata = create_attention_metadata(config, /*is_prefill=*/true);
  auto kv_cache = create_kv_cache(config);

  // Run forward pass
  LOG(INFO) << "Running forward pass...";
  auto output = attn->forward(
      positions, hidden_states, attn_metadata, kv_cache, batch_to_kv_state);

  synchronize_device();

  // Verify output shape
  LOG(INFO) << "Verifying output...";
  print_tensor_info("output", output);
  ASSERT_EQ(output.dim(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), num_tokens) << "Output batch*seq should match";
  ASSERT_EQ(output.size(1), config.dim)
      << "Output dim should match hidden_size";

  // Verify output values (from Python test log for compress_ratio=128)
  // Expected: sum=268670337024.000000, min=29952.000000, max=32640.000000
  LOG(INFO) << "Verifying output values:";
  test::verify_sum_min_max(output, 2.68670337e11f, 29952.0f, 32640.0f);

  // Verify KV cache updates (first 3 blocks)
  auto k_cache = kv_cache.get_k_cache();
  verify_kv_cache_blocks(k_cache,
                         "k_cache",
                         {{/*Block 0:*/ {47.25f, -0.141602f, 0.141602f},
                           /*Block 1:*/ {47.25f, -0.141602f, 0.141602f},
                           /*Block 2:*/ {47.25f, -0.141602f, 0.141602f}}});

  // Indexer KV cache is empty for compress_ratio != 4, skip verification

  LOG(INFO) << "DeepSeekV4Attention forward test passed!";
}

TEST_F(DeepSeekV4AttentionTest, ForwardDecodeCompressRatio4Test) {
  DeepSeekV4AttentionTestConfig config;
  config.compress_ratio = 4;
  config.batch_size = 2;
  config.seq_len = 512;

  const int64_t layer_id = 0;
  const int64_t batch_size = config.batch_size;
  const int64_t num_tokens = batch_size;
  const int64_t prefill_seq_len = 512;
  const int64_t decode_pos_start = prefill_seq_len;

  LOG(INFO) << "Testing DeepSeekV4Attention forward pass (decode)";
  LOG(INFO) << "  compress_ratio: " << config.compress_ratio;
  LOG(INFO) << "  batch_size: " << batch_size;
  LOG(INFO) << "  prefill_seq_len: " << prefill_seq_len;
  LOG(INFO) << "  num_decode_tokens: " << num_tokens;

  // Create attention module
  auto attn = create_attention_module(config, layer_id);

  // Prepare input data (Decode mode: [batch_size, dim])
  auto hidden_states =
      torch::full({num_tokens, config.dim}, config.input_val, options_);
  print_tensor_info("hidden_states", hidden_states);

  // positions: [prefill_seq_len, prefill_seq_len+1, ...]
  std::vector<int32_t> positions_vec(batch_size);
  for (int32_t i = 0; i < batch_size; ++i) {
    positions_vec[i] = static_cast<int32_t>(decode_pos_start + i);
  }
  auto positions = torch::tensor(positions_vec, int_options_);
  LOG(INFO) << "positions: [" << positions_vec[0] << ", " << positions_vec[1]
            << "]";

  auto batch_to_kv_state = create_batch_to_kv_state(batch_size);

  // Create attention metadata for decode
  int64_t num_blocks = calculate_num_blocks(config);
  std::vector<int32_t> seq_lens_vec(batch_size, prefill_seq_len + 1);
  std::vector<int32_t> q_cu_seq_lens_vec(batch_size + 1);
  for (int32_t i = 0; i <= batch_size; ++i) {
    q_cu_seq_lens_vec[i] = i;
  }

  AttentionMetadata metadata;
  metadata.q_cu_seq_lens = torch::tensor(q_cu_seq_lens_vec, int_options_);
  metadata.kv_seq_lens = torch::tensor(seq_lens_vec, int_options_);
  metadata.q_seq_lens = torch::full({batch_size}, 1, int_options_);
  metadata.block_table = torch::arange(num_blocks, int_options_)
                             .unsqueeze(0)
                             .repeat({batch_size, 1});
  metadata.max_query_len = 1;
  metadata.max_seq_len = prefill_seq_len + 1;
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  metadata.compute_dtype = "float";

  // Create KV cache and pre-fill it (simulating prefill phase)
  auto kv_cache = create_kv_cache(config);
  auto k_cache = kv_cache.get_k_cache();
  auto index_cache = kv_cache.get_index_cache();

  LOG(INFO) << "Pre-filling KV cache (simulating prefill phase)...";
  const int64_t window_size = config.window_size;
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    for (int64_t i = 0; i < std::min(window_size, prefill_seq_len); ++i) {
      k_cache[i][0][0] =
          torch::full({config.head_dim}, 0.1f + i * 0.01f, options_);
    }
  }
  LOG(INFO) << "KV cache pre-filled for " << batch_size << " sequences";

  // Run forward pass (Decode mode)
  LOG(INFO) << "Running forward pass (decode mode)...";
  auto output = attn->forward(
      positions, hidden_states, metadata, kv_cache, batch_to_kv_state);

  synchronize_device();

  // Verify output shape
  LOG(INFO) << "Verifying output...";
  print_tensor_info("output", output);
  ASSERT_EQ(output.dim(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), num_tokens) << "Output batch should match";
  ASSERT_EQ(output.size(1), config.dim)
      << "Output dim should match hidden_size";

  // Verify output values (from Python test log for decode with
  // compress_ratio=4) Expected: sum=3388997632.0, min=413696.0, max=413696.0
  LOG(INFO) << "Verifying output values:";
  test::verify_sum_min_max(output, 3388997632.0f, 413696.0f, 413696.0f);

  // Verify KV cache (first 3 blocks)
  k_cache = kv_cache.get_k_cache();
  verify_kv_cache_blocks(k_cache,
                         "k_cache",
                         {{/*Block 0:*/ {47.25f, -0.141602f, 0.140625f},
                           /*Block 1:*/ {56.25f, 0.109863f, 0.109863f},
                           /*Block 2:*/ {61.5f, 0.120117f, 0.120117f}}});

  // Verify Indexer KV cache (first 3 blocks, should all be zeros)
  index_cache = kv_cache.get_index_cache();
  verify_kv_cache_blocks(index_cache,
                         "index_cache",
                         {{/*Block 0:*/ {0.0f, 0.0f, 0.0f},
                           /*Block 1:*/ {0.0f, 0.0f, 0.0f},
                           /*Block 2:*/ {0.0f, 0.0f, 0.0f}}});

  LOG(INFO) << "DeepSeekV4Attention decode test passed!";
}

}  // namespace layer
}  // namespace xllm
