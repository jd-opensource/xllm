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

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_MLU)
#include "layers/mlu/attention.h"
#include "layers/mlu/deepseek_v2_decoder_layer_impl.h"
#elif defined(USE_CUDA)
#include "layers/cuda/attention.h"
#endif
#include "layers/common/tests/tests_utils.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

class DeepseekV2DecoderLayerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    FLAGS_enable_mla = true;  // Enable MLA for DeepSeek V2 attention
    // Base defaults from test helpers
    model_args_ = test::create_default_model_args();
    // test w8a8 only for now
    quant_args_ = test::create_default_quant_args();
    options_ = torch::TensorOptions()
                   .dtype(torch::kBFloat16)
                   .device(Device::type_torch(), 0)
                   .requires_grad(false);
    parallel_args_ = test::create_default_parallel_args(mock_process_group_);
    // Fill additional DeepSeek V3 defaults
    model_args_.model_type() = "deepseek_v3";
    model_args_.dtype() = "";  // default empty
    model_args_.vocab_size() = 129280;
    model_args_.n_layers() = 61;
    model_args_.n_heads() = 128;
    model_args_.n_kv_heads() = 128;
    model_args_.intermediate_size() = 18432;  // Ensure matches defaults
    model_args_.hidden_act() = "silu";
    model_args_.rms_norm_eps() = 1e-6f;
    model_args_.max_position_embeddings() = 163840;
    model_args_.eos_token_id() = 1;
    model_args_.bos_token_id() = 0;

    // Decoder layer specific routing between MoE and Dense
    model_args_.first_k_dense_replace() = 3;
    model_args_.moe_layer_freq() = 1;

    // MoE-related params
    model_args_.n_routed_experts() = 256;
    model_args_.n_shared_experts() = 1;
    model_args_.num_experts_per_tok() = 8;
    model_args_.moe_intermediate_size() = 2048;
    model_args_.routed_scaling_factor() = 2.5f;
    model_args_.norm_topk_prob() = true;
    model_args_.n_group() = 8;
    model_args_.topk_group() = 4;
    model_args_.scoring_func() = "sigmoid";
    model_args_.topk_method() = "noaux_tc";

    // Q/K/V dims used by DeepseekV2Attention
    model_args_.qk_nope_head_dim() = 128;
    model_args_.qk_rope_head_dim() = 64;
    model_args_.v_head_dim() = 128;
    model_args_.head_dim() = 256;  // qk_nope_head_dim + qk_rope_head_dim
    model_args_.rotary_dim() = model_args_.qk_rope_head_dim();

    // Rope scaling related
    model_args_.rope_scaling_rope_type() = "deepseek_yarn";
    // The following values may be model/export dependent; set common defaults.
    model_args_.rope_scaling_beta_fast() = 32;
    model_args_.rope_scaling_beta_slow() = 1;
    model_args_.rope_scaling_factor() = 40;
    model_args_.rope_extrapolation_factor() = 1.0f;
    model_args_.rope_scaling_mscale() = 1.0f;
    model_args_.rope_scaling_mscale_all_dim() = 1.0f;
    model_args_.rope_scaling_original_max_position_embeddings() = 4096;
    model_args_.rope_scaling_attn_factor() = 1.0f;

    // Sliding window
    model_args_.use_sliding_window() = false;
    model_args_.sliding_window() = 4096;
    model_args_.max_window_layers() = 61;

    // LORA ranks for DeepSeek-V3
    model_args_.q_lora_rank() = 1536;
    model_args_.kv_lora_rank() = 512;

    // extra parameters for DeepSeek-V3.2-Exp
    model_args_.index_head_dim() = 128;
    model_args_.index_n_heads() = 64;
    model_args_.index_topk() = 2048;

    // Build a ModelContext that the decoder requires
    context_ = ModelContext(parallel_args_, model_args_, quant_args_, options_);
  }

  // Collect registered child module names to verify module wiring
  static std::unordered_set<std::string> get_child_module_names(
      const torch::nn::Module& module) {
    std::unordered_set<std::string> names;
    for (const auto& named_child : module.named_children()) {
      names.insert(named_child.key());
    }
    return names;
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

  // Create default test weights for decoder layer (w8a8 smoothquant format)
  std::unordered_map<std::string, torch::Tensor> create_default_test_weights(
      int32_t layer_id,
      int64_t hidden_size,
      int64_t intermediate_size,
      int64_t moe_intermediate_size = -1,
      int num_routed_experts = -1) {
    std::unordered_map<std::string, torch::Tensor> weight_dict;

    // Create input_layernorm weights (float32, not quantized)
    // Shape: [hidden_size]
    auto input_norm_weight = torch::full({hidden_size}, 1.0f, options_);
    weight_dict["input_layernorm.weight"] =
        input_norm_weight.to(torch::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(options_.device()));

    // Create post_attention_layernorm weights (float32, not quantized)
    // Shape: [hidden_size]
    auto post_norm_weight = torch::full({hidden_size}, 1.0f, options_);
    weight_dict["post_attention_layernorm.weight"] =
        post_norm_weight.to(torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(options_.device()));

    // Determine if this layer uses Dense MLP or MoE
    bool use_moe = (layer_id >= model_args_.first_k_dense_replace());

    if (use_moe) {
      // Create MoE weights
      int64_t test_moe_intermediate_size =
          (moe_intermediate_size > 0) ? moe_intermediate_size
                                      : model_args_.moe_intermediate_size();
      int test_num_routed_experts = (num_routed_experts > 0)
                                        ? num_routed_experts
                                        : model_args_.n_routed_experts();

      // Create gate weights (routing layer, not quantized)
      // Shape: [num_routed_experts, hidden_size]
      auto gate_weight =
          torch::full({test_num_routed_experts, hidden_size}, 0.8f, options_);
      weight_dict["mlp.gate.weight"] = gate_weight;

      // Create e_score_correction_bias if needed
      auto e_score_correction_bias =
          torch::full({test_num_routed_experts}, 0.1f, options_);
      weight_dict["mlp.gate.e_score_correction_bias"] = e_score_correction_bias;

      // Create shared experts weights if n_shared_experts > 0
      if (model_args_.n_shared_experts() > 0) {
        // gate_proj weights
        auto shared_gate_weight = torch::full(
            {test_moe_intermediate_size, hidden_size}, 0.3f, options_);
        auto shared_gate_qweight = shared_gate_weight.to(torch::kInt8);
        auto shared_gate_scale = torch::full({test_moe_intermediate_size},
                                             0.1f,
                                             torch::TensorOptions()
                                                 .dtype(torch::kFloat32)
                                                 .device(options_.device()));
        auto shared_gate_smooth = torch::full({hidden_size},
                                              0.05f,
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(options_.device()));

        // up_proj weights
        auto shared_up_weight = torch::full(
            {test_moe_intermediate_size, hidden_size}, 0.3f, options_);
        auto shared_up_qweight = shared_up_weight.to(torch::kInt8);
        auto shared_up_scale = torch::full({test_moe_intermediate_size},
                                           0.1f,
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(options_.device()));
        auto shared_up_smooth = torch::full({hidden_size},
                                            0.05f,
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(options_.device()));

        // down_proj weights
        auto shared_down_weight = torch::full(
            {hidden_size, test_moe_intermediate_size}, 0.2f, options_);
        auto shared_down_qweight = shared_down_weight.to(torch::kInt8);
        auto shared_down_scale = torch::full({hidden_size},
                                             0.1f,
                                             torch::TensorOptions()
                                                 .dtype(torch::kFloat32)
                                                 .device(options_.device()));
        auto shared_down_smooth = torch::full({test_moe_intermediate_size},
                                              0.05f,
                                              torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(options_.device()));

        weight_dict["mlp.shared_experts.gate_proj.qweight"] =
            shared_gate_qweight;
        weight_dict["mlp.shared_experts.gate_proj.per_channel_scale"] =
            shared_gate_scale;
        weight_dict["mlp.shared_experts.gate_proj.smooth"] = shared_gate_smooth;
        weight_dict["mlp.shared_experts.up_proj.qweight"] = shared_up_qweight;
        weight_dict["mlp.shared_experts.up_proj.per_channel_scale"] =
            shared_up_scale;
        weight_dict["mlp.shared_experts.up_proj.smooth"] = shared_up_smooth;
        weight_dict["mlp.shared_experts.down_proj.qweight"] =
            shared_down_qweight;
        weight_dict["mlp.shared_experts.down_proj.per_channel_scale"] =
            shared_down_scale;
        weight_dict["mlp.shared_experts.down_proj.smooth"] = shared_down_smooth;
      }

      // Create routed experts weights
      for (int expert_id = 0; expert_id < test_num_routed_experts;
           ++expert_id) {
        std::string expert_prefix =
            "mlp.experts." + std::to_string(expert_id) + ".";

        // gate_proj weights
        auto gate_proj_weight = torch::full(
            {test_moe_intermediate_size, hidden_size}, 0.5f, options_);
        auto gate_proj_qweight = gate_proj_weight.to(torch::kInt8);
        auto gate_proj_scale = torch::full({test_moe_intermediate_size},
                                           0.1f,
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(options_.device()));
        auto gate_proj_smooth = torch::full({hidden_size},
                                            0.05f,
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(options_.device()));

        // up_proj weights
        auto up_proj_weight = torch::full(
            {test_moe_intermediate_size, hidden_size}, 0.5f, options_);
        auto up_proj_qweight = up_proj_weight.to(torch::kInt8);
        auto up_proj_scale = torch::full({test_moe_intermediate_size},
                                         0.1f,
                                         torch::TensorOptions()
                                             .dtype(torch::kFloat32)
                                             .device(options_.device()));
        auto up_proj_smooth = torch::full({hidden_size},
                                          0.05f,
                                          torch::TensorOptions()
                                              .dtype(torch::kFloat32)
                                              .device(options_.device()));

        // down_proj weights
        auto down_proj_weight = torch::full(
            {hidden_size, test_moe_intermediate_size}, 0.3f, options_);
        auto down_proj_qweight = down_proj_weight.to(torch::kInt8);
        auto down_proj_scale = torch::full({hidden_size},
                                           0.1f,
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat32)
                                               .device(options_.device()));
        auto down_proj_smooth = torch::full({test_moe_intermediate_size},
                                            0.05f,
                                            torch::TensorOptions()
                                                .dtype(torch::kFloat32)
                                                .device(options_.device()));

        weight_dict[expert_prefix + "gate_proj.qweight"] = gate_proj_qweight;
        weight_dict[expert_prefix + "gate_proj.per_channel_scale"] =
            gate_proj_scale;
        weight_dict[expert_prefix + "gate_proj.smooth"] = gate_proj_smooth;
        weight_dict[expert_prefix + "up_proj.qweight"] = up_proj_qweight;
        weight_dict[expert_prefix + "up_proj.per_channel_scale"] =
            up_proj_scale;
        weight_dict[expert_prefix + "up_proj.smooth"] = up_proj_smooth;
        weight_dict[expert_prefix + "down_proj.qweight"] = down_proj_qweight;
        weight_dict[expert_prefix + "down_proj.per_channel_scale"] =
            down_proj_scale;
        weight_dict[expert_prefix + "down_proj.smooth"] = down_proj_smooth;
      }
    } else {
      // Create Dense MLP weights
      // gate_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto gate_weight =
          torch::full({intermediate_size, hidden_size}, 5.0f, options_);
      auto gate_qweight = gate_weight.to(torch::kInt8);
      auto gate_scale = torch::full({intermediate_size},
                                    0.1f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
      auto gate_smooth = torch::full({hidden_size},
                                     0.05f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

      // up_proj weights (ColumnParallelLinear)
      // Shape: [intermediate_size, hidden_size]
      auto up_weight =
          torch::full({intermediate_size, hidden_size}, 5.0f, options_);
      auto up_qweight = up_weight.to(torch::kInt8);
      auto up_scale = torch::full({intermediate_size},
                                  0.1f,
                                  torch::TensorOptions()
                                      .dtype(torch::kFloat32)
                                      .device(options_.device()));
      auto up_smooth = torch::full({hidden_size},
                                   0.05f,
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(options_.device()));

      // down_proj weights (RowParallelLinear)
      // Shape: [hidden_size, intermediate_size]
      auto down_weight =
          torch::full({hidden_size, intermediate_size}, 3.0f, options_);
      auto down_qweight = down_weight.to(torch::kInt8);
      auto down_scale = torch::full({hidden_size},
                                    0.1f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
      auto down_smooth = torch::full({intermediate_size},
                                     0.05f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

      weight_dict["mlp.gate_proj.qweight"] = gate_qweight;
      weight_dict["mlp.gate_proj.per_channel_scale"] = gate_scale;
      weight_dict["mlp.gate_proj.smooth"] = gate_smooth;
      weight_dict["mlp.up_proj.qweight"] = up_qweight;
      weight_dict["mlp.up_proj.per_channel_scale"] = up_scale;
      weight_dict["mlp.up_proj.smooth"] = up_smooth;
      weight_dict["mlp.down_proj.qweight"] = down_qweight;
      weight_dict["mlp.down_proj.per_channel_scale"] = down_scale;
      weight_dict["mlp.down_proj.smooth"] = down_smooth;
    }

    // Create attention weights for DeepSeek V2 (MLA)
    int64_t num_heads = model_args_.n_heads();
    int64_t q_lora_rank = model_args_.q_lora_rank();
    int64_t kv_lora_rank = model_args_.kv_lora_rank();
    int64_t qk_nope_head_dim = model_args_.qk_nope_head_dim();
    int64_t qk_rope_head_dim = model_args_.qk_rope_head_dim();
    int64_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    int64_t v_head_dim = model_args_.v_head_dim();
    int64_t index_head_dim = model_args_.index_head_dim();
    int64_t index_n_heads = model_args_.index_n_heads();

    // Quantized weights (w8a8 smoothquant format)
    // o_proj weights
    auto o_proj_weight =
        torch::full({hidden_size, num_heads * v_head_dim}, 1.0f, options_);
    auto o_proj_qweight = o_proj_weight.to(torch::kInt8);
    auto o_proj_scale = torch::full({hidden_size},
                                    0.03f,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(options_.device()));
    auto o_proj_smooth = torch::full({num_heads * v_head_dim},
                                     0.03f,
                                     torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

    weight_dict["self_attn.o_proj.qweight"] = o_proj_qweight;
    weight_dict["self_attn.o_proj.per_channel_scale"] = o_proj_scale;
    weight_dict["self_attn.o_proj.smooth"] = o_proj_smooth;

    // q_b_proj weights
    auto q_b_proj_weight =
        torch::full({num_heads * qk_head_dim, q_lora_rank}, 1.0f, options_);
    auto q_b_proj_qweight = q_b_proj_weight.to(torch::kInt8);
    auto q_b_proj_scale = torch::full({num_heads * qk_head_dim},
                                      0.03f,
                                      torch::TensorOptions()
                                          .dtype(torch::kFloat32)
                                          .device(options_.device()));
    auto q_b_proj_smooth = torch::full({q_lora_rank},
                                       0.03f,
                                       torch::TensorOptions()
                                           .dtype(torch::kFloat32)
                                           .device(options_.device()));

    weight_dict["self_attn.q_b_proj.qweight"] = q_b_proj_qweight;
    weight_dict["self_attn.q_b_proj.per_channel_scale"] = q_b_proj_scale;
    weight_dict["self_attn.q_b_proj.smooth"] = q_b_proj_smooth;

    // Non-quantized weights (float32)
    // kv_b_proj.weight: [num_heads * (qk_nope_head_dim + v_head_dim),
    // kv_lora_rank]
    auto kv_b_proj_weight =
        torch::full({num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank},
                    0.02f,
                    options_);
    weight_dict["self_attn.kv_b_proj.weight"] =
        kv_b_proj_weight.to(torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(options_.device()));

    // kv_a_proj_with_mqa.weight: [kv_lora_rank + qk_rope_head_dim, hidden_size]
    auto kv_a_proj_with_mqa_weight = torch::full(
        {kv_lora_rank + qk_rope_head_dim, hidden_size}, 0.02f, options_);
    weight_dict["self_attn.kv_a_proj_with_mqa.weight"] =
        kv_a_proj_with_mqa_weight.to(torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(options_.device()));

    // q_a_proj.weight: [q_lora_rank, hidden_size]
    auto q_a_proj_weight =
        torch::full({q_lora_rank, hidden_size}, 0.02f, options_);
    weight_dict["self_attn.q_a_proj.weight"] =
        q_a_proj_weight.to(torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(options_.device()));

    // LayerNorm weights
    auto kv_a_layernorm_weight = torch::full({kv_lora_rank}, 1.0f, options_);
    weight_dict["self_attn.kv_a_layernorm.weight"] =
        kv_a_layernorm_weight.to(torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(options_.device()));

    auto q_a_layernorm_weight = torch::full({q_lora_rank}, 1.0f, options_);
    weight_dict["self_attn.q_a_layernorm.weight"] =
        q_a_layernorm_weight.to(torch::TensorOptions()
                                    .dtype(torch::kFloat32)
                                    .device(options_.device()));

    // Indexer weights (if enabled)
    if (model_args_.index_n_heads() > 0) {
      auto indexer_k_norm_bias = torch::full({index_head_dim}, 0.0f, options_);
      weight_dict["self_attn.indexer.k_norm.bias"] =
          indexer_k_norm_bias.to(torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(options_.device()));

      auto indexer_k_norm_weight =
          torch::full({index_head_dim}, 1.0f, options_);
      weight_dict["self_attn.indexer.k_norm.weight"] =
          indexer_k_norm_weight.to(torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(options_.device()));

      auto indexer_weights_proj_weight =
          torch::full({index_n_heads, hidden_size}, 0.02f, options_);
      weight_dict["self_attn.indexer.weights_proj.weight"] =
          indexer_weights_proj_weight.to(torch::TensorOptions()
                                             .dtype(torch::kFloat32)
                                             .device(options_.device()));

      auto indexer_wk_weight =
          torch::full({index_head_dim, hidden_size}, 0.02f, options_);
      weight_dict["self_attn.indexer.wk.weight"] =
          indexer_wk_weight.to(torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(options_.device()));

      auto indexer_wq_b_weight = torch::full(
          {index_n_heads * index_head_dim, q_lora_rank}, 0.02f, options_);
      weight_dict["self_attn.indexer.wq_b.weight"] =
          indexer_wq_b_weight.to(torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(options_.device()));
    }

    LOG(INFO) << "Test w8a8 smoothquant weights created successfully for layer "
              << layer_id << " (use_moe=" << use_moe << ")";

    return weight_dict;
  }

  // Helper function to create test weights with custom dimensions
  std::unordered_map<std::string, torch::Tensor> create_test_weights(
      int32_t layer_id,
      int64_t custom_hidden_size = -1,
      int64_t custom_intermediate_size = -1,
      int64_t custom_moe_intermediate_size = -1,
      int custom_num_routed_experts = -1) {
    int64_t test_hidden_size = (custom_hidden_size > 0)
                                   ? custom_hidden_size
                                   : model_args_.hidden_size();
    int64_t test_intermediate_size = (custom_intermediate_size > 0)
                                         ? custom_intermediate_size
                                         : model_args_.intermediate_size();

    return create_default_test_weights(layer_id,
                                       test_hidden_size,
                                       test_intermediate_size,
                                       custom_moe_intermediate_size,
                                       custom_num_routed_experts);
  }

  ModelArgs model_args_;
  QuantArgs quant_args_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::unique_ptr<xllm::ProcessGroup> mock_process_group_;
  ModelContext context_{};

  // Expected output for precision verification
  std::vector<float> expected_output_;
};

TEST_F(DeepseekV2DecoderLayerTest,
       ConstructorRegistersExpectedSubmodules_FirstLayer) {
  // layer_id < first_k_dense_replace → Dense MLP path inside decoder
  int32_t layer_id = 0;
  auto decoder = torch::nn::ModuleHolder<DeepseekV2DecoderLayerImpl>(
      DeepseekV2DecoderLayerImpl(context_, layer_id));

  auto child_names = get_child_module_names(*decoder);
  // Core components should be registered with these names (see implementation)
  EXPECT_TRUE(child_names.count("self_attn")) << "self_attn missing";
  EXPECT_TRUE(child_names.count("input_layernorm"))
      << "input_layernorm missing";
  EXPECT_TRUE(child_names.count("post_attention_layernorm"))
      << "post_attention_layernorm missing";
  EXPECT_TRUE(child_names.count("mlp")) << "mlp missing";
}

TEST_F(DeepseekV2DecoderLayerTest,
       ConstructorRegistersExpectedSubmodules_DenseLayer) {
  // layer_id >= first_k_dense_replace → MoE path inside decoder
  int32_t layer_id = std::max<int32_t>(5, model_args_.first_k_dense_replace());
  auto decoder = torch::nn::ModuleHolder<DeepseekV2DecoderLayerImpl>(
      DeepseekV2DecoderLayerImpl(context_, layer_id));

  auto child_names = get_child_module_names(*decoder);
  EXPECT_TRUE(child_names.count("self_attn"));
  EXPECT_TRUE(child_names.count("input_layernorm"));
  EXPECT_TRUE(child_names.count("post_attention_layernorm"));
  EXPECT_TRUE(child_names.count("mlp"));  // name is the same for MoE/Dense
}

TEST_F(DeepseekV2DecoderLayerTest, LoadStateDictTest_DenseMLP) {
  // Test loading weights into the decoder layer with Dense MLP
  int32_t layer_id = 0;  // < first_k_dense_replace, uses Dense MLP
  auto decoder = torch::nn::ModuleHolder<DeepseekV2DecoderLayerImpl>(
      DeepseekV2DecoderLayerImpl(context_, layer_id));

  // Create test weights
  auto weight_dict = create_test_weights(layer_id);

  // Load weights into the decoder
  StateDict state_dict(weight_dict);
  EXPECT_NO_THROW(decoder->load_state_dict(state_dict));
  LOG(INFO) << "Dense MLP state dict loading test passed";
}

TEST_F(DeepseekV2DecoderLayerTest, LoadStateDictTest_FusedMoE) {
  // Test loading weights into the decoder layer with FusedMoE
  int32_t layer_id = std::max<int32_t>(5, model_args_.first_k_dense_replace());
  auto decoder = torch::nn::ModuleHolder<DeepseekV2DecoderLayerImpl>(
      DeepseekV2DecoderLayerImpl(context_, layer_id));

  // Create test weights
  auto weight_dict = create_test_weights(layer_id);

  // Load weights into the decoder
  StateDict state_dict(weight_dict);
  EXPECT_NO_THROW(decoder->load_state_dict(state_dict));
  LOG(INFO) << "FusedMoE state dict loading test passed";
}

TEST_F(DeepseekV2DecoderLayerTest,
       SmoothquantPrecisionVerificationTest_DenseMLP) {
  // Test precision verification with custom input and expected output for Dense
  int32_t layer_id = 0;  // Use Dense MLP path
  const int64_t batch_size = 16;
  const int64_t seq_len = 8;
  int64_t block_size = 1;
  // 1000 is just a random value for some space
  int64_t block_num = batch_size * seq_len * block_size * 1000;

  context_ = ModelContext(parallel_args_, model_args_, quant_args_, options_);

  // Create decoder with custom dimensions
  auto decoder = torch::nn::ModuleHolder<DeepseekV2DecoderLayerImpl>(
      DeepseekV2DecoderLayerImpl(context_, layer_id));

  // Create test weights with custom dimensions
  auto weight_dict = create_test_weights(layer_id);

  // Load weights into the decoder
  StateDict state_dict(weight_dict);
  decoder->load_state_dict(state_dict);

  // Create hidden states tensor with pesodu random value
  auto hidden_states = xllm::layer::test::seeded_tensor(
      "hidden_states",
      {batch_size * seq_len, model_args_.hidden_size()},
      at::kBFloat16,
      options_.device());

  // Create positions tensor
  auto positions = torch::arange(
      0,
      batch_size * seq_len,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Build minimal ModelInputParams for prefill
  ModelInputParams input_params;
  input_params.empty_kv_cache = true;
  input_params.num_sequences = batch_size;
  input_params.q_max_seq_len = seq_len;
  input_params.kv_max_seq_len = batch_size * seq_len;

  // Create q_seq_lens (cumulative sequence lengths for queries)
  // Shape: [batch_size + 1], e.g., [0, seq_len, 2*seq_len, ...]
  input_params.q_seq_lens = torch::arange(
      0,
      (batch_size + 1) * seq_len,
      seq_len,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Create kv_seq_lens (cumulative sequence lengths for kv cache)
  // Shape: [batch_size + 1], same as q_seq_lens for prefill
  input_params.kv_seq_lens = torch::arange(
      0,
      (batch_size + 1) * seq_len,
      seq_len,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Create q_seq_lens_vec and kv_seq_lens_vec for chunked prefill check
  input_params.q_seq_lens_vec.resize(batch_size, seq_len);
  input_params.kv_seq_lens_vec.resize(batch_size, seq_len);

  // Create new_cache_slots (slot mapping for new tokens)
  input_params.new_cache_slots = torch::arange(
      1,
      batch_size * seq_len + 1,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Create block_tables (required for MLA)
  // Shape: [batch_size, max_num_batched_tokens]
  int64_t max_num_batched_tokens = batch_size * seq_len;
  input_params.block_tables = torch::zeros(
      {batch_size, max_num_batched_tokens},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Fill block_table with consecutive numbers (similar to mla_tests.cpp)
  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t start_val = b * seq_len + 1;
    int64_t end_val = start_val + seq_len;
    auto seq = torch::arange(
        start_val,
        end_val,
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
    input_params.block_tables[b].index_put_(
        {torch::indexing::Slice(0, seq_len)}, seq);
  }

  input_params = input_params.to(options_.device());

  // Build AttentionMetadata for prefill
  auto attn_metadata = AttentionMetadata::build(input_params);

  // Build KVCache with valid shapes
  // Reference: mla_tests.cpp - k_cache shape: [block_num, 1, 1,
  // qk_rope_head_dim + kv_lora_rank] index_cache shape: [block_num, 1, 1,
  // index_head_dim]
  int64_t qk_rope_head_dim = model_args_.qk_rope_head_dim();
  int64_t kv_lora_rank = model_args_.kv_lora_rank();
  int64_t index_head_dim = model_args_.index_head_dim();

  // Create KVCache tensors following mla_tests.cpp pattern
  auto k_cache = torch::zeros(
      {block_num, 1, block_size, qk_rope_head_dim + kv_lora_rank}, options_);
  auto index_cache =
      torch::zeros({block_num, 1, block_size, index_head_dim}, options_);
  KVCache kv_cache(k_cache, torch::Tensor(), index_cache);

  std::optional<torch::Tensor> residual = std::nullopt;
  auto output = decoder->forward(hidden_states,
                                 residual,
                                 positions,
                                 attn_metadata,
                                 kv_cache,
                                 input_params);

  // Synchronize device stream
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  ASSERT_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), batch_size * seq_len) << "Batch size should match";
  ASSERT_EQ(output.size(1), model_args_.hidden_size())
      << "Hidden size should match";

  // Set expected output values for precision verification (only first 5
  // elements) The expected values should be calculated based on vLLM MLU
  const int num_elements_to_check = 5;
  std::vector<float> expected_values(num_elements_to_check, 8847360.0f);

  // Extract first 5 elements from output and compare
  auto output_flat = output.flatten().to(torch::kFloat32).cpu();
  auto output_first_5 = output_flat.slice(0, 0, num_elements_to_check);

  // Create expected tensor for comparison
  auto expected_tensor = torch::tensor(
      expected_values, torch::TensorOptions().dtype(torch::kFloat32));

  // Verify precision for first 5 elements
  ASSERT_TRUE(torch::allclose(output_first_5, expected_tensor, 1e-3, 1e-4))
      << "First 5 elements do not match expected values";
}

TEST_F(DeepseekV2DecoderLayerTest, SmoothquantPrecisionVerificationTest_MoE) {
  // Test precision verification with custom input and expected output for MoE
  int32_t layer_id = std::max<int32_t>(
      5, model_args_.first_k_dense_replace());  // Use MoE path
  const int64_t batch_size = 16;
  const int64_t seq_len = 8;
  int64_t block_size = 1;
  // 1000 is just a random value for some space
  int64_t block_num = batch_size * seq_len * block_size * 1000;

  context_ = ModelContext(parallel_args_, model_args_, quant_args_, options_);

  // Create decoder with custom dimensions
  auto decoder = torch::nn::ModuleHolder<DeepseekV2DecoderLayerImpl>(
      DeepseekV2DecoderLayerImpl(context_, layer_id));

  // Create test weights with custom dimensions
  auto weight_dict = create_test_weights(layer_id);

  // Load weights into the decoder
  StateDict state_dict(weight_dict);
  decoder->load_state_dict(state_dict);

  // Create hidden states tensor with pesodu random value
  auto hidden_states = xllm::layer::test::seeded_tensor(
      "hidden_states",
      {batch_size * seq_len, model_args_.hidden_size()},
      at::kBFloat16,
      options_.device());

  // Create positions tensor
  auto positions = torch::arange(
      0,
      batch_size * seq_len,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Build minimal ModelInputParams for prefill
  ModelInputParams input_params;
  input_params.empty_kv_cache = true;
  input_params.num_sequences = batch_size;
  input_params.q_max_seq_len = seq_len;
  input_params.kv_max_seq_len = batch_size * seq_len;

  // Create q_seq_lens (cumulative sequence lengths for queries)
  // Shape: [batch_size + 1], e.g., [0, seq_len, 2*seq_len, ...]
  input_params.q_seq_lens = torch::arange(
      0,
      (batch_size + 1) * seq_len,
      seq_len,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Create kv_seq_lens (cumulative sequence lengths for kv cache)
  // Shape: [batch_size + 1], same as q_seq_lens for prefill
  input_params.kv_seq_lens = torch::arange(
      0,
      (batch_size + 1) * seq_len,
      seq_len,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Create q_seq_lens_vec and kv_seq_lens_vec for chunked prefill check
  input_params.q_seq_lens_vec.resize(batch_size, seq_len);
  input_params.kv_seq_lens_vec.resize(batch_size, seq_len);

  // Create new_cache_slots (slot mapping for new tokens)
  input_params.new_cache_slots = torch::arange(
      1,
      batch_size * seq_len + 1,
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Create block_tables (required for MLA)
  // Shape: [batch_size, max_num_batched_tokens]
  int64_t max_num_batched_tokens = batch_size * seq_len;
  input_params.block_tables = torch::zeros(
      {batch_size, max_num_batched_tokens},
      torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));

  // Fill block_table with consecutive numbers (similar to mla_tests.cpp)
  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t start_val = b * seq_len + 1;
    int64_t end_val = start_val + seq_len;
    auto seq = torch::arange(
        start_val,
        end_val,
        torch::TensorOptions().dtype(torch::kInt32).device(options_.device()));
    input_params.block_tables[b].index_put_(
        {torch::indexing::Slice(0, seq_len)}, seq);
  }

  input_params = input_params.to(options_.device());

  // Build AttentionMetadata for prefill
  auto attn_metadata = AttentionMetadata::build(input_params);

  // Build KVCache with valid shapes
  // Reference: mla_tests.cpp - k_cache shape: [block_num, 1, 1,
  // qk_rope_head_dim + kv_lora_rank] index_cache shape: [block_num, 1, 1,
  // index_head_dim]
  int64_t qk_rope_head_dim = model_args_.qk_rope_head_dim();
  int64_t kv_lora_rank = model_args_.kv_lora_rank();
  int64_t index_head_dim = model_args_.index_head_dim();

  // Create KVCache tensors following mla_tests.cpp pattern
  auto k_cache = torch::zeros(
      {block_num, 1, block_size, qk_rope_head_dim + kv_lora_rank}, options_);
  auto index_cache =
      torch::zeros({block_num, 1, block_size, index_head_dim}, options_);
  KVCache kv_cache(k_cache, torch::Tensor(), index_cache);

  std::optional<torch::Tensor> residual = std::nullopt;
  auto output = decoder->forward(hidden_states,
                                 residual,
                                 positions,
                                 attn_metadata,
                                 kv_cache,
                                 input_params);

  // Synchronize device stream
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  // Verify output shape
  ASSERT_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
  ASSERT_EQ(output.size(0), batch_size * seq_len) << "Batch size should match";
  ASSERT_EQ(output.size(1), model_args_.hidden_size())
      << "Hidden size should match";

  // Set expected output values for precision verification (only first 5
  // elements)
  // TODO: Fill in expected values based on vLLM MLU reference output
  const int num_elements_to_check = 5;
  std::vector<float> expected_values;
  expected_values.reserve(num_elements_to_check);
  expected_values.push_back(151.0);
  expected_values.push_back(151.0);
  expected_values.push_back(152.0);
  expected_values.push_back(152.0);
  expected_values.push_back(151.0);

  // Extract first 5 elements from output and compare
  auto output_flat = output.flatten().to(torch::kFloat32).cpu();
  auto output_first_5 = output_flat.slice(0, 0, num_elements_to_check);

  // Create expected tensor for comparison
  auto expected_tensor = torch::tensor(
      expected_values, torch::TensorOptions().dtype(torch::kFloat32));

  // Verify precision for first 5 elements
  ASSERT_TRUE(torch::allclose(output_first_5, expected_tensor, 1e-3, 1e-4))
      << "First 5 elements do not match expected values";
}

}  // namespace layer
}  // namespace xllm
