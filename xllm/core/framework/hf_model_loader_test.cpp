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
#include <fstream>

#include "core/platform/device.h"

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

class HFModelLoaderConfigTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = std::filesystem::temp_directory_path() / "hf_model_loader_test";
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
    std::filesystem::create_directories(test_dir_);
  }

  void TearDown() override {
    if (std::filesystem::exists(test_dir_)) {
      std::filesystem::remove_all(test_dir_);
    }
  }

  void write_file(const std::filesystem::path& path, const std::string& data) {
    std::ofstream file(path);
    ASSERT_TRUE(file.is_open());
    file << data;
    file.close();
  }

  std::filesystem::path test_dir_;
};

TEST_F(HFModelLoaderConfigTest, TreatsTopLevelDtypeAsTorchDtype) {
  write_file(test_dir_ / "config.json", R"json(
    {
      "model_type": "qwen3",
      "dtype": "bfloat16"
    }
  )json");
  write_file(test_dir_ / "model.safetensors", "");

  HFModelLoader loader(test_dir_.string());
  EXPECT_EQ(loader.model_args().dtype(), "bfloat16");
  EXPECT_EQ(loader.quant_args().torch_dtype(), "bfloat16");
}

}  // namespace xllm
