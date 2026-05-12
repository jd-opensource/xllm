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

#pragma once

#include <cstdint>
#include <string>

#include "core/common/macros.h"

namespace xllm {

class KVCacheConfig final {
 public:
  KVCacheConfig() = default;
  ~KVCacheConfig() = default;

  static KVCacheConfig& get_instance();

  static KVCacheConfig from_flags();
  static void reload_from_flags();

  PROPERTY(int32_t, block_size) = 128;

  PROPERTY(int64_t, max_cache_size) = 0;

  PROPERTY(double, max_memory_utilization) = 0.8;

  PROPERTY(std::string, kv_cache_dtype) = "auto";

  PROPERTY(bool, enable_prefix_cache) = true;

  PROPERTY(uint32_t, xxh3_128bits_seed) = 1024;

  PROPERTY(bool, enable_xtensor) = false;

  PROPERTY(int64_t, phy_page_granularity_size) = 2 * 1024 * 1024;
};

}  // namespace xllm
