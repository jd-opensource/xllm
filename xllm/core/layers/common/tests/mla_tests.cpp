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

#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v2_attention.h"
#include "platform/device.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {

class DeepseekMLATest : public ::testing::Test {
 protected:
  void SetUp() override {
    FLAGS_enable_mla = true;
    // Initialize default model arguments for testing
    model_args_ = create_mla_model_args();

    // Initialize w8a8 quantization arguments
    quant_args_ = test::create_default_quant_args();

    // Initialize tensor options
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Device::type_torch(), 0)
                   .requires_grad(false);

    // Create mock ProcessGroup and initialize ParallelArgs
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);

    init_test_weights();
  }

  ModelArgs create_mla_model_args() {
    ModelArgs model_args;
    model_args.q_lora_rank() = 1536;
    model_args.kv_lora_rank() = 512;
    model_args.qk_nope_head_dim() = 128;
    model_args.qk_rope_head_dim() = 64;
    model_args.v_head_dim() = 128;
    model_args.hidden_size() = 7168;
    model_args.n_heads() = 128;
    model_args.max_position_embeddings() = 163840;
    model_args.rope_theta() = 10000;
    model_args.rms_norm_eps() = 1e-06;

    // rope_scaling config
    model_args.rope_scaling_original_max_position_embeddings() = 4096;
    model_args.rope_scaling_factor() = 40;
    model_args.rope_extrapolation_factor() = 1.;
    model_args.rope_scaling_attn_factor() = 1.;
    model_args.rope_scaling_beta_fast() = 32;
    model_args.rope_scaling_beta_slow() = 1;
    model_args.rope_scaling_mscale() = 1.;
    model_args.rope_scaling_mscale_all_dim() = 1.;
    model_args.rope_scaling_rope_type() = "deepseek_yarn";

    // indexer
    model_args.index_head_dim() = 128;
    model_args.index_n_heads() = 64;
    model_args.index_topk() = 2048;

    return model_args;
  }

  void TearDown() override {
    // Clean up if needed
  }

  void init_test_weights() {
    int64_t q_lora_rank = model_args_.q_lora_rank();
    int64_t kv_lora_rank = model_args_.kv_lora_rank();
    int64_t qk_nope_head_dim = model_args_.qk_nope_head_dim();
    int64_t qk_rope_head_dim = model_args_.qk_rope_head_dim();
    int64_t index_topk = model_args_.index_topk();
    int64_t index_n_heads = model_args_.index_n_heads();
    int64_t index_head_dim = model_args_.index_head_dim();
    int64_t v_head_dim = model_args_.v_head_dim();
    int64_t hidden_size = model_args_.hidden_size();
    int64_t num_heads = model_args_.n_heads();
    int64_t max_position_embeddings = model_args_.max_position_embeddings();
    int64_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    std::unordered_map<std::string, std::vector<int64_t>> qweight_map = {
        {"model.layers.0.self_attn.o_proj.qweight",
         {hidden_size, num_heads * v_head_dim}},
        {"model.layers.0.self_attn.q_b_proj.qweight",
         {num_heads * qk_head_dim, q_lora_rank}},
    };
    std::unordered_map<std::string, std::vector<int64_t>> scale_map = {
        {"model.layers.0.self_attn.o_proj.per_channel_scale", {hidden_size}},
        {"model.layers.0.self_attn.q_b_proj.per_channel_scale",
         {num_heads * qk_head_dim}},
        {"model.layers.0.self_attn.o_proj.smooth", {num_heads * v_head_dim}},
        {"model.layers.0.self_attn.q_b_proj.smooth", {q_lora_rank}},
    };
    std::unordered_map<std::string, std::vector<int64_t>> weight_map = {
        {"model.layers.0.self_attn.indexer.k_norm.bias", {index_head_dim}},
        {"model.layers.0.self_attn.indexer.k_norm.weight", {index_head_dim}},
        {"model.layers.0.self_attn.kv_a_layernorm.weight", {kv_lora_rank}},
        {"model.layers.0.self_attn.q_a_layernorm.weight", {q_lora_rank}},
        {"model.layers.0.self_attn.indexer.weights_proj.weight",
         {index_n_heads, hidden_size}},
        {"model.layers.0.self_attn.indexer.wk.weight",
         {index_head_dim, hidden_size}},
        {"model.layers.0.self_attn.indexer.wq_b.weight",
         {index_n_heads * index_head_dim, q_lora_rank}},
        {"model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
         {kv_lora_rank + qk_rope_head_dim, hidden_size}},
        {"model.layers.0.self_attn.kv_b_proj.weight",
         {num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank}},
        {"model.layers.0.self_attn.q_a_proj.weight",
         {q_lora_rank, hidden_size}},
    };

    for (auto& [key, shape] : qweight_map) {
      auto tensor = torch::full(shape, 1, options_.dtype(torch::kInt8));
      weight_dict_.insert({key, tensor});
    }
    for (auto& [key, shape] : scale_map) {
      auto tensor = torch::full(shape, 0.03, options_.dtype(torch::kFloat32));
      weight_dict_.insert({key, tensor});
    }
    for (auto& [key, shape] : weight_map) {
      auto tensor = torch::full(shape, 0.02, options_);
      weight_dict_.insert({key, tensor});
    }
  }

  void populate_attention_metadata(AttentionMetadata& metadata,
                                   int64_t batch_size,
                                   int64_t max_query_len,
                                   int64_t max_seq_len,
                                   bool is_prefill,
                                   int64_t max_num_batched_tokens) {
    // Create q_cu_seq_lens tensor (cu_seq_q_lens)
    // shape = [batch_size + 1], typically [0, 4, 8, 12, ...] if max_query_len=4
    auto option_int = options_.dtype(torch::kInt32);
    metadata.q_cu_seq_lens = torch::arange(
        0, (batch_size + 1) * max_query_len, max_query_len, option_int);

    // Create kv_cu_seq_lens tensor
    // TODO: Define proper shape and values based on actual requirements
    metadata.kv_cu_seq_lens = torch::zeros({batch_size + 1}, option_int);

    // Create seq_lens tensor
    // Shape: [batch_size]
    metadata.kv_seq_lens = torch::full({batch_size}, max_query_len, option_int);

    // Create block_table tensor directly assigned to metadata
    metadata.block_table =
        torch::zeros({batch_size, max_num_batched_tokens}, option_int);

    // Fill each batch with consecutive numbers
    for (int64_t b = 0; b < batch_size; ++b) {
      int64_t start_val = b * max_query_len + 1;
      int64_t end_val = start_val + max_query_len;
      // Generate sequence [start_val, ..., end_val-1]
      auto seq = torch::arange(start_val, end_val, option_int);
      metadata.block_table[b].index_put_(
          {torch::indexing::Slice(0, max_query_len)}, seq);
    }

    // Create slot_mapping tensor directly assigned to metadata
    metadata.slot_mapping =
        torch::arange(1, batch_size * max_query_len + 1, option_int);

    metadata.max_query_len = max_query_len;
    metadata.max_seq_len = max_seq_len;
    metadata.compute_dtype = "half";
    metadata.is_prefill = is_prefill;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;

  // Helper to create a mock ProcessGroup for testing
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;

  std::unordered_map<std::string, torch::Tensor> weight_dict_;

  // Expected output for precision verification
  std::vector<float> expected_output_;
};

TEST_F(DeepseekMLATest, PrefillTest) {
  auto deepseek_mla =
      DeepseekV2Attention(model_args_, quant_args_, parallel_args_, options_);

  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  deepseek_mla->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  // Create input tensors
  int64_t batch_size = 2;
  int64_t max_query_len = 5;
  int64_t block_num = 100;
  int64_t num_tokens = batch_size * max_query_len;
  int64_t hidden_size = model_args_.hidden_size();
  auto k_cache = torch::zeros(
      {block_num,
       1,
       1,
       model_args_.qk_rope_head_dim() + model_args_.kv_lora_rank()},
      options_);
  auto index_cache =
      torch::zeros({block_num, 1, 1, model_args_.index_head_dim()}, options_);
  KVCache kv_cache(k_cache, torch::Tensor(), index_cache);

  // Create metadata object and populate it
  AttentionMetadata metadata;
  populate_attention_metadata(metadata,
                              batch_size,
                              max_query_len,
                              model_args_.max_position_embeddings(),
                              /*is_prefill=*/true,
                              num_tokens);
  auto hidden_states = torch::full({num_tokens, hidden_size}, 0.05, options_);
  auto positions = torch::arange(max_query_len, options_.dtype(torch::kInt32))
                       .repeat({batch_size});
  auto output = deepseek_mla(positions, hidden_states, metadata, kv_cache);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output is not all zeros (weights were loaded)
  auto output_cpu = output.cpu();
  auto output_sum = torch::sum(output_cpu).item<float>();
  ASSERT_NE(output_sum, 0.0f)
      << "Output should not be all zeros after loading weights, output_sum: "
      << output_sum;

  std::vector<float> result(output_cpu.numel(), 3.0312);
  test::verify_precision(output_cpu, result);
  LOG(INFO) << "Prefill test passed - output avg: "
            << output_sum / output_cpu.numel();
}

TEST_F(DeepseekMLATest, DecoderTest) {
  auto deepseek_mla =
      DeepseekV2Attention(model_args_, quant_args_, parallel_args_, options_);

  std::string prefix = "model.layers.0.self_attn.";
  StateDict state_dict(weight_dict_, prefix);
  deepseek_mla->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  // Create input tensors
  int64_t batch_size = 1;
  int64_t max_query_len = 1;
  int64_t block_num = 100;
  int64_t num_tokens = batch_size * max_query_len;
  int64_t hidden_size = model_args_.hidden_size();
  auto k_cache =
      torch::ones({block_num,
                   1,
                   1,
                   model_args_.qk_rope_head_dim() + model_args_.kv_lora_rank()},
                  options_);
  auto index_cache =
      torch::ones({block_num, 1, 1, model_args_.index_head_dim()}, options_);
  KVCache kv_cache(k_cache, torch::Tensor(), index_cache);

  // Create metadata object and populate it
  AttentionMetadata metadata;
  populate_attention_metadata(metadata,
                              batch_size,
                              max_query_len,
                              model_args_.max_position_embeddings(),
                              /*is_prefill=*/false,
                              num_tokens);
  auto hidden_states = torch::full({num_tokens, hidden_size}, 0.05, options_);
  auto positions = torch::arange(max_query_len, options_.dtype(torch::kInt32))
                       .repeat({batch_size});
  auto output = deepseek_mla(positions, hidden_states, metadata, kv_cache);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output is not all zeros (weights were loaded)
  auto output_cpu = output.cpu();
  auto output_sum = torch::sum(output_cpu).item<float>();
  ASSERT_NE(output_sum, 0.0f)
      << "Output should not be all zeros after loading weights, output_sum: "
      << output_sum;

  std::vector<float> result(output_cpu.numel(), 3.0312);
  test::verify_precision(output_cpu, result);
  LOG(INFO) << "Decoder test passed - output avg: "
            << output_sum / output_cpu.numel();
}

}  // namespace layer
}  // namespace xllm
