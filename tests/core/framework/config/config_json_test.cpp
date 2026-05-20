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

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"

namespace xllm {
namespace {

inline constexpr std::string_view kInlineConfig = R"json({
  "block_size": 16,
  "max_memory_utilization": 0.5,
  "enable_prefix_cache": false,
  "max_tokens_per_batch": 8192,
  "max_seqs_per_batch": 64
})json";

class ConfigJsonFileFlagGuard final {
 public:
  explicit ConfigJsonFileFlagGuard(const std::string& config_json_file)
      : old_config_json_file_(FLAGS_config_json_file) {
    FLAGS_config_json_file = config_json_file;
  }

  ~ConfigJsonFileFlagGuard() { FLAGS_config_json_file = old_config_json_file_; }

 private:
  std::string old_config_json_file_;
};

TEST(ConfigJsonTest, FromJsonUsesParsedOverrides) {
  const JsonReader json = config::parse_json_string(kInlineConfig);

  KVCacheConfig kv_cache_config;
  kv_cache_config.from_flags();
  kv_cache_config.from_json(json);

  SchedulerConfig scheduler_config;
  scheduler_config.from_flags();
  scheduler_config.from_json(json);

  EXPECT_EQ(kv_cache_config.block_size(), 16);
  EXPECT_DOUBLE_EQ(kv_cache_config.max_memory_utilization(), 0.5);
  EXPECT_FALSE(kv_cache_config.enable_prefix_cache());
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 8192);
  EXPECT_EQ(scheduler_config.max_seqs_per_batch(), 64);

  EXPECT_EQ(kv_cache_config.kv_cache_dtype(), "auto");
  EXPECT_EQ(scheduler_config.max_decode_token_per_sequence(), 256);
}

TEST(ConfigJsonTest, InitializeLoadsConfigJsonFileFromFlag) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() / "xllm_config_json_test.json";
  {
    std::ofstream config_file(config_path);
    config_file << kInlineConfig;
  }

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 16);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 8192);

  std::filesystem::remove(config_path);
}

TEST(ConfigJsonTest, MissingJsonFileKeepsFlagDefaults) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() /
      "xllm_config_json_test_missing.json";
  std::filesystem::remove(config_path);

  ConfigJsonFileFlagGuard flag_guard(config_path.string());

  KVCacheConfig kv_cache_config;
  kv_cache_config.initialize();

  SchedulerConfig scheduler_config;
  scheduler_config.initialize();

  EXPECT_EQ(kv_cache_config.block_size(), 128);
  EXPECT_DOUBLE_EQ(kv_cache_config.max_memory_utilization(), 0.8);
  EXPECT_EQ(scheduler_config.max_tokens_per_batch(), 10240);
  EXPECT_EQ(scheduler_config.max_seqs_per_batch(), 1024);
}

}  // namespace
}  // namespace xllm
