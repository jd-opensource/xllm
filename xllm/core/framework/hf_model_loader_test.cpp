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

#include <filesystem>

#include "core/platform/device.h"
#include "core/util/model_config_utils.h"
#if defined(USE_NPU)
#include "models/model_registry.h"
#endif

namespace xllm {

TEST(HFModelLoaderTest, Qwen35BackendAwareModelTypeSelection) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text"
      },
      "vision_config": {
        "model_type": "qwen3_5"
      }
    }
  )json"));

  const auto fake_model_path = std::filesystem::path("/tmp/Qwen3.5-27B");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path), "qwen3_5_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "vlm"), "qwen3_5_vl");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "llm"),
            "qwen3_5_text");
}

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

TEST(HFModelLoaderTest, Qwen35TextModelArgsKeepTextTypeAndMropeConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_text");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text",
        "rope_parameters": {
          "mrope_interleaved": true,
          "mrope_section": [11, 11, 10],
          "partial_rotary_factor": 0.25,
          "rope_theta": 10000000
        }
      },
      "vision_config": {
        "model_type": "qwen3_5"
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_text");
  EXPECT_EQ(args.rope_scaling_mrope_section(),
            (std::vector<int64_t>{11, 11, 10}));
  EXPECT_TRUE(args.rope_scaling_mrope_interleaved());
}
#endif

}  // namespace xllm
