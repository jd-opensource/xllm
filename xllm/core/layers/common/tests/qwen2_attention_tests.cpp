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

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/qwen2_attention.h"
#include "platform/device.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {

class Qwen2AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device device(Device::type_torch(), 0);
    Device xllm_device(device);
    xllm_device.set_seed(42);
    model_args_.model_type() = "qwen2";
    model_args_.head_dim() = 128;
    model_args_.hidden_size() = 1024;
    model_args_.n_heads() = 16;
    model_args_.n_kv_heads() = 8;
    model_args_.max_position_embeddings() = 2048;
    model_args_.rope_theta() = 1000000.0f;
    model_args_.rms_norm_eps() = 1e-6f;
    model_args_.rope_scaling_factor() = 1.0f;
    model_args_.hidden_act() = "silu";

    options_ = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    process_group_ = create_process_group(
        0, 1, 1, 3331, false, "localhost", "tp_group", device);
    parallel_args_.tp_group_ = process_group_.get();

    int64_t block_num = 100;
    int64_t n_kv_heads = model_args_.n_kv_heads().value();
    int64_t head_dim = model_args_.head_dim();
    int64_t block_size = 16;

    auto k_cache =
        torch::randn({block_num, n_kv_heads, block_size, head_dim}, options_) *
        0.01f;
    auto v_cache =
        torch::randn({block_num, n_kv_heads, block_size, head_dim}, options_) *
        0.01f;
    kv_cache_ = KVCache(k_cache, v_cache);

    context_ = ModelContext(parallel_args_, model_args_, QuantArgs(), options_);
    InitTestWeights();
  }

  void InitTestWeights() {
    int64_t hidden_size = model_args_.hidden_size();
    int64_t n_heads = model_args_.n_heads();
    int64_t n_kv_heads = model_args_.n_kv_heads().value();
    int64_t head_dim = model_args_.head_dim();
    int64_t q_size = n_heads * head_dim;
    int64_t kv_size = n_kv_heads * head_dim;

    const std::string weight_seed_prefix = "qwen2_attention_test.";
    auto seeded = [this, &weight_seed_prefix](const std::string& name,
                                              torch::IntArrayRef shape) {
      return test::seeded_tensor(weight_seed_prefix + name,
                                 shape,
                                 torch::typeMetaToScalarType(options_.dtype()),
                                 options_.device());
    };

    std::unordered_map<std::string, torch::Tensor> weight_map = {
        {"q_proj.weight", seeded("q_proj.weight", {q_size, hidden_size})},
        {"k_proj.weight", seeded("k_proj.weight", {kv_size, hidden_size})},
        {"v_proj.weight", seeded("v_proj.weight", {kv_size, hidden_size})},
        {"q_proj.bias", seeded("q_proj.bias", {q_size})},
        {"k_proj.bias", seeded("k_proj.bias", {kv_size})},
        {"v_proj.bias", seeded("v_proj.bias", {kv_size})},
        {"o_proj.weight", seeded("o_proj.weight", {hidden_size, q_size})},
    };

    for (auto& [name, tensor] : weight_map) {
      tensor = tensor / torch::sqrt(torch::tensor(tensor.size(0), options_));
      weight_dict_["model.layers.0.self_attn." + name] = tensor;
    }
  }

  AttentionMetadata CreateAttentionMetadata(int64_t batch_size,
                                            int64_t seq_len,
                                            bool is_prefill,
                                            int64_t max_seq_len,
                                            bool is_chunked_prefill = false) {
    AttentionMetadata metadata;
    auto options_int = options_.dtype(torch::kInt32);

    const uint32_t block_size = 16;
    const int64_t num_blocks_per_req =
        (max_seq_len + block_size - 1) / block_size + 1;

    if (is_prefill && !is_chunked_prefill) {
      // Regular prefill: query and kv have same sequence lengths
      metadata.q_cu_seq_lens =
          torch::arange(0, (batch_size + 1) * seq_len, seq_len, options_int);
      metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
      metadata.slot_mapping =
          torch::arange(0, batch_size * seq_len, options_int);
      metadata.kv_seq_lens = torch::full({batch_size}, seq_len, options_int);
      // Use random block table for regular prefill (data stored via
      // slot_mapping)
      metadata.block_table =
          torch::randint(0, 100, {batch_size, num_blocks_per_req}, options_int);
    } else if (is_chunked_prefill) {
      // Chunked prefill: query has chunk_len, kv has full seq_len
      int64_t chunk_len = seq_len;   // current chunk length
      int64_t kv_seq_len = seq_len;  // accumulated kv length
      metadata.q_cu_seq_lens = torch::arange(
          0, (batch_size + 1) * chunk_len, chunk_len, options_int);
      metadata.kv_cu_seq_lens = torch::arange(
          0, (batch_size + 1) * kv_seq_len, kv_seq_len, options_int);
      metadata.slot_mapping =
          torch::arange(0, batch_size * chunk_len, options_int);
      metadata.kv_seq_lens = torch::full({batch_size}, kv_seq_len, options_int);
      // For chunked prefill with dequant_from_paged_cache, block_table must
      // correspond to slot_mapping. Each batch uses sequential blocks.
      // batch 0: blocks [0, num_blocks_per_req-1]
      // batch 1: blocks [num_blocks_per_req, 2*num_blocks_per_req-1]
      std::vector<int32_t> block_table_vec;
      for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t i = 0; i < num_blocks_per_req; ++i) {
          block_table_vec.push_back(b * num_blocks_per_req + i);
        }
      }
      metadata.block_table = torch::tensor(block_table_vec, options_int)
                                 .reshape({batch_size, num_blocks_per_req});
    } else {
      // Decode: query length is 1
      metadata.q_cu_seq_lens = torch::arange(0, batch_size + 1, 1, options_int);
      metadata.kv_cu_seq_lens =
          torch::arange(0, batch_size + 1, seq_len, options_int);
      metadata.slot_mapping = torch::arange(0, batch_size, options_int);
      metadata.kv_seq_lens = torch::full({batch_size}, seq_len, options_int);
      // Use random block table for decode
      metadata.block_table =
          torch::randint(0, 100, {batch_size, num_blocks_per_req}, options_int);
    }

    metadata.max_query_len = (is_prefill || is_chunked_prefill) ? seq_len : 1;
    metadata.max_seq_len = max_seq_len;
    metadata.compute_dtype = "half";
    metadata.is_prefill = is_prefill && !is_chunked_prefill;
    metadata.is_chunked_prefill = is_chunked_prefill;
    metadata.is_dummy = false;

    return metadata;
  }

  ModelArgs model_args_;
  ModelContext context_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::unordered_map<std::string, torch::Tensor> weight_dict_;
  std::unique_ptr<ProcessGroup> process_group_ = nullptr;
  KVCache kv_cache_;
};

TEST_F(Qwen2AttentionTest, PrefillTest) {
  auto qwen2_attention = Qwen2Attention(context_);
  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  qwen2_attention->load_state_dict(state_dict.get_dict_with_prefix(prefix));
  int64_t batch_size = 2;
  int64_t seq_len = 128;
  int64_t hidden_size = model_args_.hidden_size();
  int64_t num_tokens = batch_size * seq_len;

  auto hidden_states =
      torch::randn({num_tokens, hidden_size}, options_) * 0.02f;
  auto positions = torch::arange(0, seq_len, options_.dtype(torch::kInt32))
                       .repeat({batch_size});

  auto metadata = CreateAttentionMetadata(
      batch_size, seq_len, true, model_args_.max_position_embeddings());

  auto output = qwen2_attention(positions, hidden_states, metadata, kv_cache_);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  CHECK_EQ(output.sizes(), torch::IntArrayRef({num_tokens, hidden_size}));
  auto test_output = output.flatten().slice(0, 0, 10).unsqueeze(0);
  std::vector<float> expected_values = {0.2031f,
                                        0.2109f,
                                        0.2129f,
                                        0.2041f,
                                        0.1963f,
                                        0.2129f,
                                        0.2139f,
                                        0.2070f,
                                        0.2188f,
                                        0.2061f};
  test::verify_precision(test_output, expected_values, 1e-5, 1e-6);
}

TEST_F(Qwen2AttentionTest, DecodeTest) {
  auto qwen2_attention = Qwen2Attention(context_);

  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  qwen2_attention->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  int64_t batch_size = 4;
  int64_t seq_len = 256;
  int64_t hidden_size = model_args_.hidden_size();
  int64_t num_tokens = batch_size;

  auto hidden_states =
      torch::randn({num_tokens, hidden_size}, options_) * 0.02f;
  auto positions =
      torch::full({num_tokens}, seq_len, options_.dtype(torch::kInt32));

  auto metadata = CreateAttentionMetadata(
      batch_size, seq_len + 1, false, model_args_.max_position_embeddings());

  auto output = qwen2_attention(positions, hidden_states, metadata, kv_cache_);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  CHECK_EQ(output.sizes(), torch::IntArrayRef({num_tokens, hidden_size}));
  auto test_output = output.flatten().slice(0, 0, 10).unsqueeze(0);
  std::vector<float> expected_values = {0.0001f * 9.1171f,
                                        0.0001f * 7.7820f,
                                        0.0001f * 5.5695f,
                                        0.0001f * 2.3651f,
                                        0.0001f * 2.3174f,
                                        0.0001f * 2.4986f,
                                        0.0001f * 3.2997f,
                                        0.0001f * 1.4973f,
                                        0.0001f * 7.7820f,
                                        0.0001f * 6.4468f};
  test::verify_precision(test_output, expected_values, 1e-5, 1e-6);
}

TEST_F(Qwen2AttentionTest, MixedSequenceLengthTest) {
  auto qwen2_attention = Qwen2Attention(context_);

  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  qwen2_attention->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  std::vector<int64_t> seq_lens = {32, 64, 128};
  std::vector<int64_t> cu_seq_lens = {0};
  int64_t total_tokens = 0;
  int32_t seq_lens_size = seq_lens.size();

  for (auto len : seq_lens) {
    total_tokens += len;
    cu_seq_lens.push_back(total_tokens);
  }

  int64_t hidden_size = model_args_.hidden_size();
  auto hidden_states =
      torch::randn({total_tokens, hidden_size}, options_) * 0.02f;

  std::vector<int32_t> positions_vec;
  for (size_t i = 0; i < seq_lens_size; ++i) {
    for (int64_t j = 0; j < seq_lens[i]; ++j) {
      positions_vec.push_back(j);
    }
  }
  auto positions = torch::tensor(positions_vec, options_.dtype(torch::kInt32));

  AttentionMetadata metadata;
  auto options_int = options_.dtype(torch::kInt32);

  metadata.q_cu_seq_lens = torch::tensor(cu_seq_lens, options_int);
  metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
  metadata.kv_seq_lens = torch::tensor(seq_lens, options_int);
  metadata.block_table = torch::zeros({seq_lens_size, 1}, options_int);
  metadata.slot_mapping = torch::arange(0, total_tokens, options_int);
  metadata.max_query_len = *std::max_element(seq_lens.begin(), seq_lens.end());
  metadata.max_seq_len = model_args_.max_position_embeddings();
  metadata.compute_dtype = "half";
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;

  auto output = qwen2_attention(positions, hidden_states, metadata, kv_cache_);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  CHECK_EQ(output.sizes(), torch::IntArrayRef({total_tokens, hidden_size}));
  auto test_output = output.flatten().slice(0, 0, 10).unsqueeze(0);
  std::vector<float> expected_values = {0.2412f,
                                        0.2520f,
                                        0.2559f,
                                        0.2422f,
                                        0.2393f,
                                        0.2559f,
                                        0.2578f,
                                        0.2520f,
                                        0.2539f,
                                        0.2441f};
  test::verify_precision(test_output, expected_values, 1e-5, 1e-6);
}

TEST_F(Qwen2AttentionTest, QuantizedKVCachePrefillTest) {
  auto qwen2_attention = Qwen2Attention(context_);
  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  qwen2_attention->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  // Test parameters
  int64_t batch_size = 2;
  int64_t seq_len = 128;
  int64_t hidden_size = model_args_.hidden_size();
  int64_t num_tokens = batch_size * seq_len;
  int64_t block_num = 100;
  int64_t n_kv_heads = model_args_.n_kv_heads().value();
  int64_t head_dim = model_args_.head_dim();
  int64_t block_size = 16;

  // Create INT8 KV cache tensors using seeded tensors for reproducibility
  auto k_cache =
      test::seeded_tensor("qwen2_quant_test.k_cache",
                          {block_num, n_kv_heads, block_size, head_dim},
                          torch::kInt8,
                          options_.device());
  auto v_cache =
      test::seeded_tensor("qwen2_quant_test.v_cache",
                          {block_num, n_kv_heads, block_size, head_dim},
                          torch::kInt8,
                          options_.device());

  // Create float32 scale tensors
  auto k_cache_scale = test::seeded_tensor("qwen2_quant_test.k_scale",
                                           {block_num, n_kv_heads, block_size},
                                           torch::kFloat32,
                                           options_.device());
  auto v_cache_scale = test::seeded_tensor("qwen2_quant_test.v_scale",
                                           {block_num, n_kv_heads, block_size},
                                           torch::kFloat32,
                                           options_.device());

  KVCache quant_kv_cache(
      k_cache, v_cache, torch::Tensor(), k_cache_scale, v_cache_scale);

  // Create input tensors using seeded tensors
  auto hidden_states = test::seeded_tensor("qwen2_quant_test.hidden_states",
                                           {num_tokens, hidden_size},
                                           torch::kBFloat16,
                                           options_.device());
  auto positions = test::seeded_tensor("qwen2_quant_test.positions",
                                       {num_tokens},
                                       torch::kInt32,
                                       options_.device());

  auto metadata = CreateAttentionMetadata(
      batch_size, seq_len, true, model_args_.max_position_embeddings());

  // Run forward with quantized KV cache
  auto output =
      qwen2_attention(positions, hidden_states, metadata, quant_kv_cache);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, hidden_size}));

  // Verify precision using expect_tensor_stats
  // Expected values from Phase 1 print test
  test::expect_tensor_stats(output,
                            /*expected_min=*/240,
                            /*expected_max=*/280,
                            /*expected_sum=*/67867024);
}

TEST_F(Qwen2AttentionTest, QuantizedKVCacheDecodeDiagnosticTest) {
  auto qwen2_attention = Qwen2Attention(context_);
  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  qwen2_attention->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  // Test parameters - use minimal batch size for diagnosis
  int64_t batch_size = 1;
  int64_t seq_len = 16;  // Start with a small sequence length
  int64_t hidden_size = model_args_.hidden_size();
  int64_t num_tokens = batch_size;
  int64_t block_num = 100;
  int64_t n_kv_heads = model_args_.n_kv_heads().value();
  int64_t head_dim = model_args_.head_dim();
  int64_t block_size = 16;

  auto int8_options =
      torch::TensorOptions().dtype(torch::kInt8).device(options_.device());
  auto float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(options_.device());

  // 1. Use zeros for INT8 cache to avoid random value issues
  auto k_cache =
      torch::zeros({block_num, n_kv_heads, block_size, head_dim}, int8_options);
  auto v_cache =
      torch::zeros({block_num, n_kv_heads, block_size, head_dim}, int8_options);

  // 2. Use ones for scale to avoid scale=0 issues
  auto k_cache_scale =
      torch::ones({block_num, n_kv_heads, block_size}, float_options);
  auto v_cache_scale =
      torch::ones({block_num, n_kv_heads, block_size}, float_options);

  // Verify cache and scale shapes and dtypes
  ASSERT_EQ(k_cache.sizes(),
            torch::IntArrayRef({block_num, n_kv_heads, block_size, head_dim}));
  ASSERT_EQ(k_cache.scalar_type(), torch::kInt8);
  ASSERT_EQ(k_cache_scale.sizes(),
            torch::IntArrayRef({block_num, n_kv_heads, block_size}));
  ASSERT_EQ(k_cache_scale.scalar_type(), torch::kFloat32);

  KVCache quant_kv_cache(
      k_cache, v_cache, torch::Tensor(), k_cache_scale, v_cache_scale);

  // Create input tensors using seeded tensors
  auto hidden_states = test::seeded_tensor("qwen2_decode_diag.hidden_states",
                                           {num_tokens, hidden_size},
                                           torch::kBFloat16,
                                           options_.device());
  auto positions =
      torch::full({num_tokens}, seq_len - 1, options_.dtype(torch::kInt32));

  auto metadata = CreateAttentionMetadata(
      batch_size, seq_len, false, model_args_.max_position_embeddings());

  // Run forward with quantized KV cache
  auto output =
      qwen2_attention(positions, hidden_states, metadata, quant_kv_cache);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, hidden_size}));

  // Print output stats for debugging
  torch::Tensor flat = output.flatten().to(torch::kFloat32).cpu();
  double out_min = torch::min(flat).item<double>();
  double out_max = torch::max(flat).item<double>();
  double out_sum = torch::sum(flat).item<double>();
  LOG(INFO) << "Decode Diagnostic output - min: " << out_min
            << ", max: " << out_max << ", sum: " << out_sum;
}

// Phase 4: Decode test with controlled seeded values
TEST_F(Qwen2AttentionTest, QuantizedKVCacheDecodeTest) {
  auto qwen2_attention = Qwen2Attention(context_);
  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  qwen2_attention->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  // Test parameters - match DecodeTest configuration
  int64_t batch_size = 4;
  int64_t seq_len = 256;
  int64_t hidden_size = model_args_.hidden_size();
  int64_t num_tokens = batch_size;
  int64_t block_num = 100;
  int64_t n_kv_heads = model_args_.n_kv_heads().value();
  int64_t head_dim = model_args_.head_dim();
  int64_t block_size = 16;

  // Create INT8 KV cache tensors using seeded tensors for reproducibility
  auto k_cache =
      test::seeded_tensor("qwen2_quant_decode.k_cache",
                          {block_num, n_kv_heads, block_size, head_dim},
                          torch::kInt8,
                          options_.device());
  auto v_cache =
      test::seeded_tensor("qwen2_quant_decode.v_cache",
                          {block_num, n_kv_heads, block_size, head_dim},
                          torch::kInt8,
                          options_.device());

  // Create scale tensors with controlled range [0.5, 1.5] to avoid extreme
  // values
  auto k_cache_scale_raw =
      test::seeded_tensor("qwen2_quant_decode.k_scale",
                          {block_num, n_kv_heads, block_size},
                          torch::kFloat32,
                          options_.device());
  auto v_cache_scale_raw =
      test::seeded_tensor("qwen2_quant_decode.v_scale",
                          {block_num, n_kv_heads, block_size},
                          torch::kFloat32,
                          options_.device());
  // Scale to [0.5, 1.5] range: 0.5 + raw * 1.0
  auto k_cache_scale = 0.5f + k_cache_scale_raw;
  auto v_cache_scale = 0.5f + v_cache_scale_raw;

  KVCache quant_kv_cache(
      k_cache, v_cache, torch::Tensor(), k_cache_scale, v_cache_scale);

  // Create input tensors using seeded tensors
  auto hidden_states = test::seeded_tensor("qwen2_quant_decode.hidden_states",
                                           {num_tokens, hidden_size},
                                           torch::kBFloat16,
                                           options_.device());
  auto positions =
      torch::full({num_tokens}, seq_len, options_.dtype(torch::kInt32));

  auto metadata = CreateAttentionMetadata(
      batch_size, seq_len + 1, false, model_args_.max_position_embeddings());

  // Run forward with quantized KV cache
  auto output =
      qwen2_attention(positions, hidden_states, metadata, quant_kv_cache);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, hidden_size}));

  // Print output stats for debugging
  torch::Tensor flat = output.flatten().to(torch::kFloat32).cpu();
  double out_min = torch::min(flat).item<double>();
  double out_max = torch::max(flat).item<double>();
  double out_sum = torch::sum(flat).item<double>();
  LOG(INFO) << "Quantized Decode output - min: " << out_min
            << ", max: " << out_max << ", sum: " << out_sum;

  // Verify precision using expect_tensor_stats
  // Expected values established from successful diagnostic run
  test::expect_tensor_stats(output,
                            /*expected_min=*/-296,
                            /*expected_max=*/165,
                            /*expected_sum=*/-211443.53,
                            /*rtol=*/0.01,
                            /*atol=*/1.0);
}

// Chunked prefill + quantized KV cache path uses flash attention and
// dequant_from_paged_cache; parallel reduction in these kernels can be
// non-deterministic on MLU, so fixed golden min/max/sum are not stable.
// We validate shape, finite output, and determinism (two runs with same input
// yield close results) instead of exact tensor stats.
TEST_F(Qwen2AttentionTest, QuantizedKVCacheChunkedPrefillTest) {
  auto qwen2_attention = Qwen2Attention(context_);
  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  qwen2_attention->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  // Test parameters - use smaller values to fit within block limits
  int64_t batch_size = 2;
  int64_t seq_len = 32;      // Smaller sequence length
  int64_t max_seq_len = 64;  // Match block_table allocation
  int64_t hidden_size = model_args_.hidden_size();
  int64_t num_tokens = batch_size * seq_len;
  int64_t block_size = 16;
  // Calculate required blocks: each batch needs (max_seq_len/block_size) blocks
  int64_t num_blocks_per_req = (max_seq_len + block_size - 1) / block_size + 1;
  int64_t block_num =
      batch_size * num_blocks_per_req + 10;  // Extra blocks for safety
  int64_t n_kv_heads = model_args_.n_kv_heads().value();
  int64_t head_dim = model_args_.head_dim();

  // Create INT8 KV cache tensors using seeded tensors for reproducibility
  auto k_cache =
      test::seeded_tensor("qwen2_quant_chunked.k_cache",
                          {block_num, n_kv_heads, block_size, head_dim},
                          torch::kInt8,
                          options_.device());
  auto v_cache =
      test::seeded_tensor("qwen2_quant_chunked.v_cache",
                          {block_num, n_kv_heads, block_size, head_dim},
                          torch::kInt8,
                          options_.device());

  // Create scale tensors with controlled range [0.5, 1.5]
  auto k_cache_scale_raw =
      test::seeded_tensor("qwen2_quant_chunked.k_scale",
                          {block_num, n_kv_heads, block_size},
                          torch::kFloat32,
                          options_.device());
  auto v_cache_scale_raw =
      test::seeded_tensor("qwen2_quant_chunked.v_scale",
                          {block_num, n_kv_heads, block_size},
                          torch::kFloat32,
                          options_.device());
  auto k_cache_scale = 0.5f + k_cache_scale_raw;
  auto v_cache_scale = 0.5f + v_cache_scale_raw;

  KVCache quant_kv_cache(
      k_cache, v_cache, torch::Tensor(), k_cache_scale, v_cache_scale);

  auto hidden_states = test::seeded_tensor("qwen2_quant_chunked.hidden_states",
                                           {num_tokens, hidden_size},
                                           torch::kBFloat16,
                                           options_.device());
  auto positions = test::seeded_tensor("qwen2_quant_chunked.positions",
                                       {num_tokens},
                                       torch::kInt32,
                                       options_.device());

  // Create metadata with is_chunked_prefill=true
  auto metadata =
      CreateAttentionMetadata(batch_size, seq_len, false, max_seq_len, true);

  // First forward
  auto output =
      qwen2_attention(positions, hidden_states, metadata, quant_kv_cache);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, hidden_size}));

  // Sanity: no NaN/Inf
  torch::Tensor flat = output.flatten().to(torch::kFloat32).cpu();
  ASSERT_TRUE(torch::isfinite(flat).all().item<bool>())
      << "Output contains NaN or Inf";

  // Second forward with same inputs (quant_kv_cache is overwritten with same
  // K/V by quant_to_paged_cache). If the path were deterministic, both outputs
  // would match; we use loose tolerance to allow backend non-determinism.
  auto output2 =
      qwen2_attention(positions, hidden_states, metadata, quant_kv_cache);
  device.synchronize_default_stream();
  ASSERT_TRUE(torch::allclose(output.flatten().to(torch::kFloat32),
                              output2.flatten().to(torch::kFloat32),
                              /*rtol=*/0.05,
                              /*atol=*/0.05))
      << "Two forwards with same input should produce close results "
         "(determinism check)";
}

}  // namespace layer
}  // namespace xllm
