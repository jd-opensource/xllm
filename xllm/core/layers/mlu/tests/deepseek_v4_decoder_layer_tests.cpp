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

#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/common/tests/tests_utils.h"
#include "layers/mlu/deepseek_v4/deepseek_v4_decoder_layer_impl.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

namespace {

void verify_sum_min_max(const torch::Tensor& output,
                        float expected_sum,
                        float expected_min,
                        float expected_max,
                        float rtol = 1e-2) {
  torch::Tensor output_fp32 = output.to(torch::kFloat32).flatten();
  float actual_sum = torch::sum(output_fp32).item<float>();
  float actual_min = torch::min(output_fp32).item<float>();
  float actual_max = torch::max(output_fp32).item<float>();

  EXPECT_NEAR(actual_sum, expected_sum, std::abs(expected_sum) * rtol + 1.0f);
  EXPECT_NEAR(actual_min, expected_min, std::abs(expected_min) * rtol + 0.1f);
  EXPECT_NEAR(actual_max, expected_max, std::abs(expected_max) * rtol + 0.1f);
}

}  // namespace

struct DeepSeekV4DecoderLayerTestConfig {
  // Model architecture
  int64_t hidden_size = 4096;        // dim
  int64_t intermediate_size = 2048;  // moe_inter_dim
  int64_t n_heads = 64;              // n_heads
  int64_t head_dim = 512;            // head_dim
  int64_t q_lora_rank = 1024;        // q_lora_rank
  int64_t rope_head_dim = 64;        // rope_head_dim
  int64_t o_groups = 8;              // o_groups
  int64_t o_lora_rank = 1024;        // o_lora_rank

  // Attention parameters
  int64_t window_size = 128;                // window_size
  int64_t compress_ratio = 4;               // compress_ratios[2] for layer_id=2
  int64_t max_position_embeddings = 65536;  // original_seq_len
  double rope_theta = 10000.0;              // rope_theta
  double compress_rope_theta = 160000.0;    // compress_rope_theta

  // MoE parameters
  int64_t n_routed_experts = 16;    // n_routed_experts
  int64_t n_shared_experts = 1;     // n_shared_experts
  int64_t num_experts_per_tok = 6;  // n_activated_experts
  int64_t n_group = 8;
  int64_t topk_group = 4;
  float routed_scaling_factor = 1.5f;         // route_scale
  std::string scoring_func = "sqrtsoftplus";  // score_func
  std::string topk_method = "noaux_tc";

  // HyperConnection parameters
  int64_t hc_mult = 4;             // hc_mult
  int64_t hc_sinkhorn_iters = 20;  // hc_sinkhorn_iters
  float hc_eps = 1e-6f;            // hc_eps

  // Indexer parameters
  int64_t index_n_heads = 64;    // index_n_heads
  int64_t index_head_dim = 128;  // index_head_dim
  int64_t index_topk = 512;      // index_topk

  // Other parameters
  float norm_eps = 1e-6f;        // norm_eps
  int64_t n_hash_layers = 3;     // n_hash_layers
  int64_t vocab_size = 129280;   // vocab_size
  int64_t cached_state_num = 8;  // cached_state_num

  // Test input parameters (will be overridden per test case)
  int64_t batch_size = 2;
  int64_t seq_len = 64;

  // Weight initialization values
  float weight_val1 = 0.1f;  // for weight parameters
  float weight_val2 = 0.2f;  // for bias and other parameters
  float weight_val3 = 0.3f;  // for ape/hc parameters
  float input_val = 0.01f;   // for input hidden_states
};

class DeepSeekV4DecoderLayerTest : public ::testing::Test {
 protected:
  torch::TensorOptions options_;
  torch::TensorOptions options_fp32_;
  torch::TensorOptions int_options_;
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

    mock_process_group_ = std::make_unique<test::MockProcessGroup>(
        torch::Device(Device::type_torch(), 0));
    parallel_args_ = ParallelArgs(0, 1, mock_process_group_.get());
    parallel_args_.tp_group_ = mock_process_group_.get();
    parallel_args_.single_rank_group_ = mock_process_group_.get();
    parallel_args_.sp_group_ = mock_process_group_.get();
  }

  // Calculate num_blocks for KV cache
  int64_t calculate_num_blocks(const DeepSeekV4DecoderLayerTestConfig& config) {
    int64_t max_blocks_attention =
        config.window_size +
        config.max_position_embeddings / config.compress_ratio;
    int64_t max_blocks_indexer = config.seq_len / config.compress_ratio + 100;
    return std::max(max_blocks_attention, max_blocks_indexer);
  }

  // Create ModelArgs from test config
  ModelArgs create_model_args(const DeepSeekV4DecoderLayerTestConfig& config,
                              int64_t layer_id) {
    ModelArgs args;
    args.model_type() = "deepseek_v4";
    args.hidden_size() = config.hidden_size;
    args.intermediate_size() = config.intermediate_size;
    args.n_heads() = config.n_heads;
    args.head_dim() = config.head_dim;
    args.q_lora_rank() = config.q_lora_rank;
    args.rotary_dim() = config.rope_head_dim;
    args.n_group() = config.o_groups;
    args.kv_lora_rank() = config.o_lora_rank;
    args.sliding_window() = config.window_size;
    args.max_position_embeddings() = config.max_position_embeddings;
    args.rope_theta() = static_cast<float>(config.rope_theta);
    args.rope_scaling_high_freq_factor() =
        static_cast<float>(config.compress_rope_theta);
    args.rope_scaling_factor() = 4.0f;
    args.rope_scaling_beta_fast() = 32;
    args.rope_scaling_beta_slow() = 1;
    args.rope_extrapolation_factor() = 1.0f;
    args.rope_scaling_original_max_position_embeddings() =
        config.max_position_embeddings;
    args.rms_norm_eps() = config.norm_eps;

    // MoE parameters
    args.n_routed_experts() = config.n_routed_experts;
    args.n_shared_experts() = config.n_shared_experts;
    args.num_experts_per_tok() = config.num_experts_per_tok;
    args.moe_intermediate_size() = config.intermediate_size;
    args.routed_scaling_factor() = config.routed_scaling_factor;
    args.norm_topk_prob() = true;
    args.n_group() = config.n_group;
    args.topk_group() = config.topk_group;
    args.scoring_func() = config.scoring_func;
    args.topk_method() = config.topk_method;
    args.hidden_act() = "silu";

    // HyperConnection parameters
    args.hc_mult() = config.hc_mult;
    args.hc_sinkhorn_iters() = config.hc_sinkhorn_iters;
    args.hc_eps() = config.hc_eps;

    // Indexer parameters
    args.index_n_heads() = config.index_n_heads;
    args.index_head_dim() = config.index_head_dim;
    args.index_topk() = config.index_topk;

    // Other parameters
    args.n_hash_layers() = config.n_hash_layers;
    args.vocab_size() = config.vocab_size;

    args.rope_scaling() = config.compress_ratio;

    return args;
  }

  // Create weight dictionary for DeepSeekV4DecoderLayer
  std::unordered_map<std::string, torch::Tensor> create_decoder_layer_weights(
      const DeepSeekV4DecoderLayerTestConfig& config,
      int64_t layer_id) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    const int64_t hidden_size = config.hidden_size;
    const int64_t intermediate_size = config.intermediate_size;
    const int64_t num_local_heads = config.n_heads;
    const int64_t o_local_groups = config.o_groups;
    const int64_t hc_mult = config.hc_mult;
    const int64_t hc_dim = hc_mult * hidden_size;
    const int64_t mix_hc = (2 + hc_mult) * hc_mult;  // For HyperConnectionPre

    // uses: val1=0.1, val2=0.2, val3=0.3
    const float val1 = config.weight_val1;  // for weights: 0.1
    const float val2 = config.weight_val2;  // for bias: 0.2
    const float val3 = config.weight_val3;  // for ape/hc: 0.3

    // weight init uses val1 * 0.1 for non-expert weights = 0.01
    const float weight_init = val1 * 0.1f;  // 0.01

    // HyperConnection weights for attention (all float32, use val3 for hc)
    // hc_fn/hc_base/hc_scale all use val3 (0.3)
    weight_dict["hc_attn_pre.hc_fn"] =
        torch::full({mix_hc, hc_dim}, val3, options_fp32_);
    weight_dict["hc_attn_pre.hc_base"] =
        torch::full({mix_hc}, val3, options_fp32_);
    weight_dict["hc_attn_pre.hc_scale"] = torch::full({3}, val3, options_fp32_);

    // HyperConnection weights for FFN (all float32, use val3 for hc)
    weight_dict["hc_ffn_pre.hc_fn"] =
        torch::full({mix_hc, hc_dim}, val3, options_fp32_);
    weight_dict["hc_ffn_pre.hc_base"] =
        torch::full({mix_hc}, val3, options_fp32_);
    weight_dict["hc_ffn_pre.hc_scale"] = torch::full({3}, val3, options_fp32_);

    // Normalization weights
    weight_dict["attn_norm.weight"] =
        torch::full({hidden_size}, weight_init, options_fp32_);
    weight_dict["ffn_norm.weight"] =
        torch::full({hidden_size}, weight_init, options_fp32_);

    // Attention weights (DeepSeekV4Attention)
    // Query projection weights (LoRA)
    // Linear weights use bfloat16 with weight_init (0.01), norms use bfloat16
    // (not fp32!)
    if (config.q_lora_rank > 0) {
      weight_dict["attn.wq_a.weight"] =
          torch::full({config.q_lora_rank, hidden_size}, weight_init, options_);
      weight_dict["attn.q_norm.weight"] =
          torch::full({config.q_lora_rank}, weight_init, options_);
      weight_dict["attn.wq_b.weight"] =
          torch::full({num_local_heads * config.head_dim, config.q_lora_rank},
                      weight_init,
                      options_);
    }

    // KV projection weights - Linear weights use bfloat16, norms use bfloat16
    weight_dict["attn.wkv.weight"] =
        torch::full({config.head_dim, hidden_size}, weight_init, options_);
    weight_dict["attn.kv_norm.weight"] =
        torch::full({config.head_dim}, weight_init, options_);

    // Output projection weights (bfloat16)
    weight_dict["attn.wo_a.weight"] =
        torch::full({o_local_groups * config.o_lora_rank,
                     num_local_heads * config.head_dim / config.o_groups},
                    weight_init,
                    options_);
    weight_dict["attn.wo_b.weight"] =
        torch::full({hidden_size, o_local_groups * config.o_lora_rank},
                    weight_init,
                    options_);

    // Attention sink (float32, use val2)
    weight_dict["attn.attn_sink"] =
        torch::full({num_local_heads}, val2, options_fp32_);

    // Compressor weights (only if compress_ratio > 0)
    if (config.compress_ratio > 0) {
      int64_t coeff = (config.compress_ratio == 4) ? 2 : 1;
      int64_t kv_dim = config.head_dim * coeff;

      // Compressor weights: ape uses float32 with val3 (0.3),
      // wkv/wgate use float32 with weight_init (0.01), norm uses bfloat16
      weight_dict["attn.compressor.ape"] =
          torch::full({config.compress_ratio, kv_dim}, val3, options_fp32_);
      weight_dict["attn.compressor.wkv.weight"] =
          torch::full({kv_dim, hidden_size}, weight_init, options_fp32_);
      weight_dict["attn.compressor.wgate.weight"] =
          torch::full({kv_dim, hidden_size}, weight_init, options_fp32_);
      weight_dict["attn.compressor.norm.weight"] =
          torch::full({config.head_dim}, weight_init, options_);

      // Indexer weights (only if compress_ratio == 4)
      if (config.compress_ratio == 4) {
        weight_dict["attn.indexer.wq_b.weight"] = torch::full(
            {config.index_n_heads * config.index_head_dim, config.q_lora_rank},
            weight_init,
            options_);
        weight_dict["attn.indexer.weights_proj.weight"] = torch::full(
            {config.index_n_heads, hidden_size}, weight_init, options_);

        int64_t indexer_kv_dim = config.index_head_dim * coeff;

        // Indexer compressor: ape uses val3, wkv/wgate use weight_init, norm
        // uses bfloat16
        weight_dict["attn.indexer.compressor.ape"] = torch::full(
            {config.compress_ratio, indexer_kv_dim}, val3, options_fp32_);
        weight_dict["attn.indexer.compressor.wkv.weight"] = torch::full(
            {indexer_kv_dim, hidden_size}, weight_init, options_fp32_);
        weight_dict["attn.indexer.compressor.wgate.weight"] = torch::full(
            {indexer_kv_dim, hidden_size}, weight_init, options_fp32_);
        weight_dict["attn.indexer.compressor.norm.weight"] =
            torch::full({config.index_head_dim}, weight_init, options_);
      }
    }

    // MoE weights (FusedMoE) - all use bfloat16 for non-quantized weights
    // Gate weights (routing layer) - use float32
    weight_dict["ffn.gate.weight"] = torch::full(
        {config.n_routed_experts, hidden_size}, weight_init, options_fp32_);

    // Gate routing weights: tid2eid (hash-based) or bias (standard topk)
    // Determine if hash-based routing is used based on layer_id
    bool use_hash = (layer_id < config.n_hash_layers);
    if (use_hash) {
      // tid2eid: [vocab_size, top_k] - token-to-expert mapping
      // Fill with random integers in [0, num_experts) using generator with
      // fixed seed 42
      auto gen = at::make_generator<at::CPUGeneratorImpl>(42);
      // Always generate on CPU, then move to target device (for test
      // determinism)
      auto cpu_tid2eid = torch::randint(
          0,
          config.n_routed_experts,
          {config.vocab_size, config.num_experts_per_tok},
          gen,
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
      auto tid2eid = cpu_tid2eid.to(options_.device());
      weight_dict["ffn.gate.tid2eid"] = tid2eid;
    } else {
      // bias: [num_experts] - score correction bias, fill with val2 (0.2)
      weight_dict["ffn.gate.bias"] =
          torch::full({config.n_routed_experts}, val2, options_fp32_);
    }

    // Shared experts weights if n_shared_experts > 0
    if (config.n_shared_experts > 0) {
      // Using weight_init (0.01) for shared expert
      weight_dict["ffn.shared_experts.gate_proj.weight"] =
          torch::full({intermediate_size, hidden_size}, weight_init, options_);
      weight_dict["ffn.shared_experts.up_proj.weight"] =
          torch::full({intermediate_size, hidden_size}, weight_init, options_);
      weight_dict["ffn.shared_experts.down_proj.weight"] =
          torch::full({hidden_size, intermediate_size}, weight_init, options_);
    }

    // Routed experts weights (aligned with fused_moe_tests.cpp naming)
    // expert_0: 0.01, expert_1: 0.02, expert_2: 0.03, ...
    for (int64_t expert_id = 0; expert_id < config.n_routed_experts;
         ++expert_id) {
      float expert_init = val1 * 0.1f + expert_id * 0.01f;
      std::string expert_prefix =
          "ffn.experts." + std::to_string(expert_id) + ".";

      weight_dict[expert_prefix + "gate_proj.weight"] =
          torch::full({intermediate_size, hidden_size}, expert_init, options_);
      weight_dict[expert_prefix + "up_proj.weight"] =
          torch::full({intermediate_size, hidden_size}, expert_init, options_);
      weight_dict[expert_prefix + "down_proj.weight"] =
          torch::full({hidden_size, intermediate_size}, expert_init, options_);
    }

    LOG(INFO) << "Created decoder layer weights with " << weight_dict.size()
              << " tensors for layer " << layer_id;
    return weight_dict;
  }

  // Create AttentionMetadata for testing
  AttentionMetadata create_attention_metadata(
      const DeepSeekV4DecoderLayerTestConfig& config,
      bool is_prefill) {
    AttentionMetadata metadata;
    int64_t batch_size = config.batch_size;
    int64_t seq_len = config.seq_len;
    int64_t num_blocks = calculate_num_blocks(config);

    // q_cu_seq_lens: cumulative sequence lengths [0, seq_len, 2*seq_len, ...]
    std::vector<int64_t> cu_seq_lens_vec(batch_size + 1);
    for (int64_t i = 0; i <= batch_size; ++i) {
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
  KVCache create_kv_cache(const DeepSeekV4DecoderLayerTestConfig& config) {
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

    IndexedKVCacheTensors cache_tensors;
    cache_tensors.kv_cache_tensors.key_cache = key_cache;
    cache_tensors.kv_cache_tensors.value_cache = value_cache;
    cache_tensors.index_cache = index_cache;
    return KVCache(cache_tensors);
  }

  // Create ModelInputParams for testing
  ModelInputParams create_model_input_params(
      const DeepSeekV4DecoderLayerTestConfig& config,
      bool is_prefill) {
    ModelInputParams input_params;
    int64_t batch_size = config.batch_size;
    int64_t seq_len = config.seq_len;

    if (is_prefill) {
      input_params.batch_forward_type = BatchForwardType::PREFILL;
    } else {
      input_params.batch_forward_type = BatchForwardType::DECODE;
    }

    input_params.num_sequences = batch_size;
    input_params.q_max_seq_len = is_prefill ? seq_len : 1;
    input_params.kv_max_seq_len = seq_len;

    return input_params;
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

  void synchronize_device() {
    xllm::Device device(options_.device());
    device.synchronize_default_stream();
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
};

// Test forward pass in prefill mode
TEST_F(DeepSeekV4DecoderLayerTest, ForwardPrefillTest) {
  DeepSeekV4DecoderLayerTestConfig config;
  config.batch_size = 2;
  config.seq_len = 64;
  config.input_val = 0.01f;

  const int64_t layer_id = 2;
  const int64_t batch_size = config.batch_size;
  const int64_t seq_len = config.seq_len;
  const int64_t num_tokens = batch_size * seq_len;

  // Create decoder layer
  auto model_args = create_model_args(config, layer_id);
  QuantArgs quant_args;
  ModelContext context(parallel_args_, model_args, quant_args, options_);

  auto decoder = torch::nn::ModuleHolder<DeepSeekV4DecoderLayerImpl>(
      DeepSeekV4DecoderLayerImpl(context, layer_id, config.cached_state_num));

  // Load weights (initialized with val1=0.1, val2=0.2, val3=0.3)
  auto weight_dict = create_decoder_layer_weights(config, layer_id);
  StateDict state_dict(weight_dict);
  decoder->load_state_dict(state_dict);

  // Prepare input data
  auto hidden_states =
      torch::full({num_tokens, config.hc_mult, config.hidden_size},
                  config.input_val,
                  options_);
  print_tensor_info("hidden_states (prefill)", hidden_states);

  // positions: [0, 1, 2, ..., seq_len-1] repeated for batch
  auto positions = create_prefill_positions(seq_len, batch_size);

  // input_ids: [0, 1, 2, ..., num_tokens-1] for hash-based routing
  // Since layer_id=2 < n_hash_layers=3, hash-based routing is used
  auto input_ids = torch::arange(num_tokens, int_options_);
  LOG(INFO) << "input_ids shape: " << input_ids.sizes();

  auto batch_to_kv_state = create_batch_to_kv_state(batch_size);

  // Create attention metadata and KV cache
  auto attn_metadata = create_attention_metadata(config, /*is_prefill=*/true);
  auto kv_cache = create_kv_cache(config);

  LOG(INFO) << "  key_cache shape: " << kv_cache.get_k_cache().sizes();
  LOG(INFO) << "  index_cache shape: " << kv_cache.get_index_cache().sizes();

  // Create model input params
  auto input_params = create_model_input_params(config, /*is_prefill=*/true);

  // Run forward pass
  LOG(INFO) << "Running forward pass...";
  std::optional<torch::Tensor> residual = std::nullopt;
  auto output = decoder->forward(hidden_states,
                                 residual,
                                 positions,
                                 attn_metadata,
                                 kv_cache,
                                 input_params,
                                 batch_to_kv_state,
                                 input_ids);

  synchronize_device();

  // Verify output shape
  LOG(INFO) << "Verifying output...";
  print_tensor_info("output (prefill)", output);
  CHECK_EQ(output.dim(), 3)
      << "Output should be 3D tensor [tokens, hc_mult, dim]";
  CHECK_EQ(output.size(0), num_tokens) << "Output batch*seq should match";
  CHECK_EQ(output.size(1), config.hc_mult) << "Output hc_mult should match";
  CHECK_EQ(output.size(2), config.hidden_size)
      << "Output dim should match hidden_size";

  // vLLM MLU output (prefill):
  //   sum: 25682509824.0
  //   min: 2336.0
  //   max: 28928.0
  verify_sum_min_max(output, 25682509824.0f, 2336.0f, 28928.0f);
}

// Test forward pass in decode mode
TEST_F(DeepSeekV4DecoderLayerTest, ForwardDecodeTest) {
  DeepSeekV4DecoderLayerTestConfig config;
  config.batch_size = 4;
  config.seq_len = 128;
  config.input_val = 0.02f;

  const int64_t layer_id = 2;
  const int64_t batch_size = config.batch_size;
  const int64_t prefill_seq_len = config.seq_len;
  const int64_t num_decode_tokens = batch_size;  // Decode: 1 token per sequence

  // Create decoder layer
  auto model_args = create_model_args(config, layer_id);
  QuantArgs quant_args;
  ModelContext context(parallel_args_, model_args, quant_args, options_);

  auto decoder = torch::nn::ModuleHolder<DeepSeekV4DecoderLayerImpl>(
      DeepSeekV4DecoderLayerImpl(context, layer_id, config.cached_state_num));

  // Load weights (initialized with val1=0.1, val2=0.2, val3=0.3)
  auto weight_dict = create_decoder_layer_weights(config, layer_id);
  StateDict state_dict(weight_dict);
  decoder->load_state_dict(state_dict);

  // ========== Step 1: Run prefill phase to populate KV cache ==========
  LOG(INFO) << "Step 1: Running prefill phase to populate KV cache...";

  int64_t num_prefill_tokens = batch_size * prefill_seq_len;
  // Input shape: [tokens, hc_mult, dim]
  auto hidden_states_prefill =
      torch::full({num_prefill_tokens, config.hc_mult, config.hidden_size},
                  0.01f,
                  options_);
  auto positions_prefill =
      create_prefill_positions(prefill_seq_len, batch_size);

  // input_ids for hash-based routing
  auto input_ids_prefill = torch::arange(num_prefill_tokens, int_options_);

  auto batch_to_kv_state_prefill = create_batch_to_kv_state(batch_size);
  auto attn_metadata_prefill =
      create_attention_metadata(config, /*is_prefill=*/true);
  auto kv_cache = create_kv_cache(config);
  auto input_params_prefill =
      create_model_input_params(config, /*is_prefill=*/true);

  // Run prefill to populate KV cache
  std::optional<torch::Tensor> residual = std::nullopt;
  auto _ = decoder->forward(hidden_states_prefill,
                            residual,
                            positions_prefill,
                            attn_metadata_prefill,
                            kv_cache,
                            input_params_prefill,
                            batch_to_kv_state_prefill,
                            input_ids_prefill);
  synchronize_device();
  LOG(INFO) << "  Prefill phase completed, KV cache populated";

  // ========== Step 2: Run decode phase ==========
  LOG(INFO) << "Step 2: Running decode phase...";

  // Prepare decode input data (single token per sequence)
  auto hidden_states_decode =
      torch::full({num_decode_tokens, config.hc_mult, config.hidden_size},
                  config.input_val,
                  options_);
  print_tensor_info("hidden_states (decode)", hidden_states_decode);

  // positions: [seq_len-1, seq_len-1, ...] for each sequence
  auto positions_decode =
      torch::full({batch_size}, prefill_seq_len - 1, int_options_);

  // input_ids for hash-based routing (decode phase)
  auto input_ids_decode = torch::arange(num_decode_tokens, int_options_);

  auto batch_to_kv_state_decode = create_batch_to_kv_state(batch_size);

  // Create attention metadata for decode
  int64_t num_blocks = calculate_num_blocks(config);
  std::vector<int64_t> seq_lens_vec(batch_size, prefill_seq_len);
  std::vector<int64_t> q_cu_seq_lens_vec(batch_size + 1);
  for (int64_t i = 0; i <= batch_size; ++i) {
    q_cu_seq_lens_vec[i] = i;
  }

  AttentionMetadata metadata_decode;
  metadata_decode.q_cu_seq_lens =
      torch::tensor(q_cu_seq_lens_vec, int_options_);
  metadata_decode.kv_seq_lens = torch::tensor(seq_lens_vec, int_options_);
  metadata_decode.q_seq_lens = torch::full({batch_size}, 1, int_options_);
  metadata_decode.block_table = torch::arange(num_blocks, int_options_)
                                    .unsqueeze(0)
                                    .repeat({batch_size, 1});
  metadata_decode.max_query_len = 1;
  metadata_decode.max_seq_len = prefill_seq_len;
  metadata_decode.is_prefill = false;
  metadata_decode.is_chunked_prefill = false;
  metadata_decode.compute_dtype = "float";

  // Create model input params for decode
  auto input_params_decode =
      create_model_input_params(config, /*is_prefill=*/false);

  // Run forward pass (Decode mode)
  auto output_decode = decoder->forward(hidden_states_decode,
                                        residual,
                                        positions_decode,
                                        metadata_decode,
                                        kv_cache,
                                        input_params_decode,
                                        batch_to_kv_state_decode,
                                        input_ids_decode);

  synchronize_device();

  // Verify output shape
  LOG(INFO) << "Verifying output...";
  print_tensor_info("output (decode)", output_decode);
  CHECK_EQ(output_decode.dim(), 3)
      << "Output should be 3D tensor [batch, hc_mult, dim]";
  CHECK_EQ(output_decode.size(0), num_decode_tokens)
      << "Output batch should match";
  CHECK_EQ(output_decode.size(1), config.hc_mult)
      << "Output hc_mult should match";
  CHECK_EQ(output_decode.size(2), config.hidden_size)
      << "Output dim should match hidden_size";

  // vLLM MLU output (decode):
  //   sum: 476839936.0
  //   min: 2352.0
  //   max: 13376.0
  verify_sum_min_max(output_decode, 476839936.0f, 2352.0f, 13376.0f);
}

}  // namespace layer
}  // namespace xllm
