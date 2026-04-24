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

#include "hf_model_loader.h"

#include <gtest/gtest.h>

#include "core/platform/device.h"
#if defined(USE_NPU) || defined(USE_MLU)
#include "models/model_registry.h"
#endif

namespace xllm {

TEST(HFModelLoaderTest, LoadCompressedTensorsFp8StaticConfig) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "config_groups": {
          "group_0": {
            "input_activations": {
              "dynamic": false,
              "num_bits": 8,
              "type": "float"
            },
            "weights": {
              "num_bits": 8,
              "type": "float"
            }
          }
        },
        "ignore": [
          "lm_head",
          "model.layers.1.mlp.down_proj"
        ],
        "quant_method": "compressed-tensors"
      }
    }
  )json"));

  QuantArgs quant_args;
  if (Device::type_str() == "cuda") {
    ASSERT_TRUE(load_quant_cfg(reader, quant_args));
    EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
    EXPECT_EQ(quant_args.bits(), 8);
    EXPECT_EQ(quant_args.moe_weight_bits(), 8);
    EXPECT_FALSE(quant_args.activation_dynamic());
    ASSERT_EQ(quant_args.ignored_modules().size(), 2);
    EXPECT_EQ(quant_args.ignored_modules()[0], "lm_head");
    EXPECT_EQ(quant_args.ignored_modules()[1], "model.layers.1.mlp.down_proj");
  }
}

TEST(HFModelLoaderTest, KeepLegacyFp8ConfigUnchanged) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "activation_scheme": "static",
        "quant_method": "fp8"
      }
    }
  )json"));

  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));
  EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
  EXPECT_FALSE(quant_args.activation_dynamic());
}

#if defined(USE_NPU)
TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromDenseConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_mtp");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5",
      "text_config": {
        "mtp_num_hidden_layers": 1,
        "layer_types": ["linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(args.n_layers(), 1);
  ASSERT_EQ(args.layer_types().size(), 1);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
}

TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromMoeConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_moe_mtp");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5_moe",
      "text_config": {
        "mtp_num_hidden_layers": 2,
        "layer_types": ["linear_attention", "linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_moe_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 2);
  EXPECT_EQ(args.n_layers(), 2);
  ASSERT_EQ(args.layer_types().size(), 2);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
  EXPECT_EQ(args.layer_types()[1], "full_attention");
}
#endif

#if defined(USE_MLU)
TEST(HFModelLoaderTest, DeepSeekV4ModelArgsMatchFlashConfig) {
  auto loader = ModelRegistry::get_model_args_loader("deepseek_v4");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "deepseek_v4",
      "torch_dtype": "bfloat16",
      "vocab_size": 129280,
      "hidden_size": 4096,
      "moe_intermediate_size": 2048,
      "num_hidden_layers": 43,
      "hidden_act": "silu",
      "num_attention_heads": 64,
      "num_key_value_heads": 1,
      "n_routed_experts": 256,
      "n_shared_experts": 1,
      "num_experts_per_tok": 6,
      "scoring_func": "sqrtsoftplus",
      "routed_scaling_factor": 1.5,
      "q_lora_rank": 1024,
      "head_dim": 512,
      "qk_rope_head_dim": 64,
      "o_groups": 8,
      "o_lora_rank": 1024,
      "sliding_window": 128,
      "max_position_embeddings": 1048576,
      "rope_theta": 10000,
      "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn"
      },
      "index_head_dim": 128,
      "index_n_heads": 64,
      "index_topk": 512,
      "hc_mult": 4,
      "hc_sinkhorn_iters": 20,
      "hc_eps": 1e-6,
      "num_hash_layers": 3,
      "swiglu_limit": 10.0,
      "tie_word_embeddings": false,
      "eos_token_id": 1,
      "bos_token_id": 0,
      "compress_rope_theta": 160000,
      "compress_ratios": [
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0
      ]
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "deepseek_v4");
  EXPECT_EQ(args.dtype(), "bfloat16");
  EXPECT_EQ(args.hidden_size(), 4096);
  EXPECT_EQ(args.n_layers(), 43);
  EXPECT_EQ(args.n_heads(), 64);
  ASSERT_TRUE(args.n_kv_heads().has_value());
  EXPECT_EQ(args.n_kv_heads().value(), 1);
  EXPECT_EQ(args.hc_mult(), 4);
  EXPECT_EQ(args.num_experts_per_tok(), 6);
  EXPECT_EQ(args.index_topk(), 512);
  EXPECT_EQ(args.n_hash_layers(), 3);
  ASSERT_TRUE(args.swiglu_limit().has_value());
  EXPECT_FLOAT_EQ(args.swiglu_limit().value(), 10.0f);
  EXPECT_FLOAT_EQ(args.compress_rope_theta(), 160000.0f);
  ASSERT_EQ(args.compress_ratios().size(), 44);
  EXPECT_EQ(args.compress_ratios()[2], 4);
  EXPECT_EQ(args.compress_ratios()[3], 128);
  EXPECT_EQ(args.compress_ratios()[43], 0);
  EXPECT_EQ(args.o_groups(), 8);
  EXPECT_EQ(args.o_lora_rank(), 1024);
  EXPECT_EQ(args.rotary_dim(), 64);
  EXPECT_EQ(args.qk_rope_head_dim(), 64);
  EXPECT_EQ(args.kv_lora_rank(), 512);
}
#endif

}  // namespace xllm
