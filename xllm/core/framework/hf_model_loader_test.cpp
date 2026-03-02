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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace xllm {
namespace {

class TempModelDir {
 public:
  TempModelDir() {
    char dir_template[] = "/tmp/hf_model_loader_test_XXXXXX";
    char* created_dir = mkdtemp(dir_template);
    CHECK(created_dir != nullptr);
    path_ = created_dir;
  }

  ~TempModelDir() { std::filesystem::remove_all(path_); }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, const std::string& content) {
  std::ofstream file(path);
  ASSERT_TRUE(file.is_open()) << "Failed to open " << path;
  file << content;
  file.close();
}

TEST(HFModelLoaderTest, DetectsDeepseekMixedW4A8FromConfig) {
  TempModelDir model_dir;
  write_file(model_dir.path() / "config.json", R"json(
{
  "model_type": "deepseek_v32",
  "torch_dtype": "bfloat16",
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
)json");
  write_file(model_dir.path() / "dummy.safetensors", "");

  HFModelLoader loader(model_dir.path());

  EXPECT_EQ(loader.quant_args().bits(), 8);
  EXPECT_EQ(loader.quant_args().moe_weight_bits(), 4);
  EXPECT_EQ(loader.quant_args().group_size(), 128);
}

}  // namespace
}  // namespace xllm
