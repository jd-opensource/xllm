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

    std::unordered_map<std::string, torch::Tensor> weight_map = {
        {"q_proj.weight", torch::randn({q_size, hidden_size}, options_)},
        {"k_proj.weight", torch::randn({kv_size, hidden_size}, options_)},
        {"v_proj.weight", torch::randn({kv_size, hidden_size}, options_)},
        {"q_proj.bias", torch::randn({q_size}, options_)},
        {"k_proj.bias", torch::randn({kv_size}, options_)},
        {"v_proj.bias", torch::randn({kv_size}, options_)},
        {"o_proj.weight", torch::randn({hidden_size, q_size}, options_)},
    };

    for (auto& [name, tensor] : weight_map) {
      tensor = tensor / torch::sqrt(torch::tensor(tensor.size(0), options_));
      weight_dict_["model.layers.0.self_attn." + name] = tensor;
    }
  }

  AttentionMetadata CreateAttentionMetadata(int64_t batch_size,
                                            int64_t seq_len,
                                            bool is_prefill,
                                            int64_t max_seq_len) {
    AttentionMetadata metadata;
    auto options_int = options_.dtype(torch::kInt32);

    if (is_prefill) {
      metadata.q_cu_seq_lens =
          torch::arange(0, (batch_size + 1) * seq_len, seq_len, options_int);
      metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
      metadata.slot_mapping =
          torch::arange(0, batch_size * seq_len, options_int);
    } else {
      metadata.q_cu_seq_lens = torch::arange(0, batch_size + 1, 1, options_int);
      metadata.kv_cu_seq_lens =
          torch::arange(0, batch_size + 1, seq_len, options_int);
      metadata.slot_mapping = torch::arange(0, batch_size, options_int);
    }

    const uint32_t block_size = 16;
    const int64_t num_blocks_per_req =
        (max_seq_len + block_size - 1) / block_size + 1;
    metadata.kv_seq_lens = torch::full({batch_size}, seq_len, options_int);
    metadata.block_table =
        torch::randint(0, 100, {batch_size, num_blocks_per_req}, options_int);

    metadata.max_query_len = is_prefill ? seq_len : 1;
    metadata.max_seq_len = max_seq_len;
    metadata.compute_dtype = "half";
    metadata.is_prefill = is_prefill;
    metadata.is_chunked_prefill = false;
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
  std::vector<float> expected_values = {0.0917969,
                                        0.00613403,
                                        -0.0490723,
                                        0.0766602,
                                        -0.0327148,
                                        -0.0371094,
                                        -0.0466309,
                                        0.0253906,
                                        -0.0541992,
                                        0.0424805};
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
  std::vector<float> expected_values = {-0.000411987,
                                        0.000113487,
                                        0.000747681,
                                        0.000123024,
                                        -0.00124359,
                                        0.000873566,
                                        0.000455856,
                                        0.00135803,
                                        -0.00119781,
                                        -0.000526428};
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
  std::vector<float> expected_values = {0.0145264,
                                        0.0148315,
                                        0.0098877,
                                        0.0314941,
                                        -0.0291748,
                                        -0.0197754,
                                        -0.0522461,
                                        0.032959,
                                        -0.0488281,
                                        -0.0219727};
  test::verify_precision(test_output, expected_values, 1e-5, 1e-6);
}

}  // namespace layer
}  // namespace xllm
