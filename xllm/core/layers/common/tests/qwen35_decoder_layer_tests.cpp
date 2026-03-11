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
#include "framework/parallel_state/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/qwen35_decoder_layer.h"
#include "platform/device.h"
#include "tests_utils.h"

namespace xllm {
namespace layer {

class Qwen35DecoderLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device device(Device::type_torch(), 0);
    Device xllm_device(device);
    xllm_device.set_seed(42);

    model_args_.model_type() = "qwen35";
    model_args_.head_dim() = 128;
    model_args_.hidden_size() = 1024;
    model_args_.n_heads() = 16;
    model_args_.n_kv_heads() = 8;
    model_args_.max_position_embeddings() = 2048;
    model_args_.rope_theta() = 1000000.0f;
    model_args_.rms_norm_eps() = 1e-6f;
    model_args_.rope_scaling_factor() = 1.0f;
    model_args_.hidden_act() = "silu";
    model_args_.intermediate_size() = 4096;
    model_args_.n_layers() = 4;

    model_args_.linear_conv_kernel_dim() = 4;
    model_args_.linear_key_head_dim() = 128;
    model_args_.linear_value_head_dim() = 128;
    model_args_.linear_num_key_heads() = 16;
    model_args_.linear_num_value_heads() = 32;
    model_args_.full_attention_interval() = 4;

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
    int64_t intermediate_size = model_args_.intermediate_size();

    const std::string weight_seed_prefix = "qwen35_decoder_test.";
    auto seeded = [this, &weight_seed_prefix](const std::string& name,
                                              torch::IntArrayRef shape) {
      return test::seeded_tensor(weight_seed_prefix + name,
                                 shape,
                                 torch::typeMetaToScalarType(options_.dtype()),
                                 options_.device());
    };

    std::unordered_map<std::string, torch::Tensor> weight_map = {
        {"self_attn.q_proj.weight", seeded("q_proj.weight", {q_size, hidden_size})},
        {"self_attn.k_proj.weight", seeded("k_proj.weight", {kv_size, hidden_size})},
        {"self_attn.v_proj.weight", seeded("v_proj.weight", {kv_size, hidden_size})},
        {"self_attn.o_proj.weight", seeded("o_proj.weight", {hidden_size, q_size})},
        {"mlp.gate_proj.weight", seeded("gate_proj.weight", {intermediate_size, hidden_size})},
        {"mlp.up_proj.weight", seeded("up_proj.weight", {intermediate_size, hidden_size})},
        {"mlp.down_proj.weight", seeded("down_proj.weight", {hidden_size, intermediate_size})},
        {"input_layernorm.weight", seeded("input_norm.weight", {hidden_size})},
        {"post_attention_layernorm.weight", seeded("post_norm.weight", {hidden_size})},
    };

    for (auto& [name, tensor] : weight_map) {
      tensor = tensor / torch::sqrt(torch::tensor(tensor.size(0), options_));
      weight_dict_["model.layers.0." + name] = tensor;
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
      metadata.q_cu_seq_lens =
          torch::arange(0, (batch_size + 1) * seq_len, seq_len, options_int);
      metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
      metadata.q_seq_lens = std::vector<int32_t>(batch_size, seq_len);
      metadata.kv_seq_lens = std::vector<int32_t>(batch_size, seq_len);
    } else {
      metadata.q_cu_seq_lens =
          torch::arange(0, batch_size + 1, 1, options_int);
      metadata.kv_cu_seq_lens =
          torch::arange(0, batch_size + 1, 1, options_int);
      metadata.q_seq_lens = std::vector<int32_t>(batch_size, 1);
      metadata.kv_seq_lens = std::vector<int32_t>(batch_size, seq_len);
    }

    metadata.is_prefill = is_prefill;
    metadata.is_chunked_prefill = is_chunked_prefill;
    metadata.max_seq_len = max_seq_len;

    metadata.block_tables =
        torch::arange(0, batch_size * num_blocks_per_req, 1, options_int)
            .reshape({batch_size, num_blocks_per_req});
    metadata.slot_mapping =
        torch::arange(0, batch_size, 1, options_int);

    return metadata;
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_;
  torch::TensorOptions options_;
  std::shared_ptr<ProcessGroup> process_group_;
  KVCache kv_cache_;
  ModelContext context_;
  std::unordered_map<std::string, torch::Tensor> weight_dict_;
};

TEST_F(Qwen35DecoderLayerTest, BasicForwardTest) {
  auto layer = Qwen35DecoderLayer(context_, 0);
  std::string prefix = "model.layers.0.";
  StateDict state_dict(weight_dict_, prefix);
  layer->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  int64_t batch_size = 2;
  int64_t seq_len = 16;
  int64_t hidden_size = model_args_.hidden_size();
  int64_t num_tokens = batch_size * seq_len;

  auto hidden_states = test::seeded_tensor("qwen35_test.hidden_states",
                                           {num_tokens, hidden_size},
                                           torch::kBFloat16,
                                           options_.device());
  auto positions = test::seeded_tensor("qwen35_test.positions",
                                       {num_tokens},
                                       torch::kInt32,
                                       options_.device());

  auto metadata = CreateAttentionMetadata(
      batch_size, seq_len, true, model_args_.max_position_embeddings());

  std::optional<torch::Tensor> residual = std::nullopt;

  auto output = layer(hidden_states, residual, positions, metadata, kv_cache_, {});

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, hidden_size}));
  EXPECT_FALSE(output.isnan().any().item<bool>());
  EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(Qwen35DecoderLayerTest, LinearAttentionLayerTest) {
  auto layer = Qwen35DecoderLayer(context_, 1);
  std::string prefix = "model.layers.1.";
  StateDict state_dict(weight_dict_, prefix);
  layer->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  int64_t batch_size = 2;
  int64_t seq_len = 16;
  int64_t hidden_size = model_args_.hidden_size();
  int64_t num_tokens = batch_size * seq_len;

  auto hidden_states = test::seeded_tensor("qwen35_linear_test.hidden_states",
                                           {num_tokens, hidden_size},
                                           torch::kBFloat16,
                                           options_.device());
  auto positions = test::seeded_tensor("qwen35_linear_test.positions",
                                       {num_tokens},
                                       torch::kInt32,
                                       options_.device());

  auto metadata = CreateAttentionMetadata(
      batch_size, seq_len, true, model_args_.max_position_embeddings());

  std::optional<torch::Tensor> residual = std::nullopt;

  auto output = layer(hidden_states, residual, positions, metadata, kv_cache_, {});

  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  ASSERT_EQ(output.sizes(), torch::IntArrayRef({num_tokens, hidden_size}));
}

}  // namespace layer
}  // namespace xllm
