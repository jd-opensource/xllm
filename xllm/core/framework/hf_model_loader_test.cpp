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

#include <filesystem>
#include <string>

namespace xllm {
namespace {

std::filesystem::path get_fixture_dir(const std::string& name) {
  return std::filesystem::path(__FILE__).parent_path() / "testdata" /
         "hf_model_loader" / name;
}

TEST(HFModelLoaderTest, DetectsDeepseekMixedW4A8FromConfig) {
  HFModelLoader loader(get_fixture_dir("deepseek_w4a8"));

  EXPECT_EQ(loader.quant_args().bits(), 8);
  EXPECT_EQ(loader.quant_args().moe_weight_bits(), 4);
  EXPECT_EQ(loader.quant_args().group_size(), 128);
}

TEST(HFModelLoaderTest, KeepsDefaultMoeWeightBitsWhenBitsMissing) {
  HFModelLoader loader(get_fixture_dir("deepseek_w4a8_no_bits"));

  EXPECT_EQ(loader.quant_args().bits(), 0);
  EXPECT_EQ(loader.quant_args().moe_weight_bits(), 8);
  EXPECT_EQ(loader.quant_args().group_size(), 128);
}

}  // namespace
}  // namespace xllm
