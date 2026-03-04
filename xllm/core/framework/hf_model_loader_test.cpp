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

#include "core/framework/hf_model_loader.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>

namespace xllm {
namespace {

TEST(HFModelLoaderTest, DetectsDeepseekMixedW4A8FromConfig) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
{
  "quantization_config": {
    "quant_method": "smoothquant",
    "bits": 8,
    "group_size": 128,
    "weight_precision": "int8",
    "activation_precision": "int8",
    "only_expert_per_group": true,
    "expert_weight_precision": "int4",
    "experts_weight_bits": 4,
    "expert_activation_precision": "int8"
  }
}
)json"));
  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));

  EXPECT_EQ(quant_args.bits(), 8);
  EXPECT_EQ(quant_args.moe_weight_bits(), 4);
  EXPECT_EQ(quant_args.group_size(), 128);
}

TEST(HFModelLoaderTest, KeepsDefaultMoeWeightBitsWhenBitsMissing) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
{
  "quantization_config": {
    "quant_method": "smoothquant",
    "group_size": 128
  }
}
)json"));
  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));

  EXPECT_EQ(quant_args.bits(), 0);
  EXPECT_EQ(quant_args.moe_weight_bits(), 8);
  EXPECT_EQ(quant_args.group_size(), 128);
}

}  // namespace
}  // namespace xllm
