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

#include "core/framework/config/kv_cache_config.h"

#include "core/common/global_flags.h"

DEFINE_int32(block_size,
             128,
             "Number of slots per kv cache block. Default is 128.");

DEFINE_int64(max_cache_size,
             0,
             "Max gpu memory size for kv cache. Default is 0, which means "
             "cache size is caculated by available memory.");

DEFINE_double(max_memory_utilization,
              0.8,
              "The fraction of GPU memory to be used for model inference, "
              "including model weights and kv cache.");

DEFINE_string(
    kv_cache_dtype,
    "auto",
    "KV cache data type for quantization. \"auto\" (default): KV "
    "cache dtype aligns with model dtype (no quantization). "
    "\"int8\": Enables INT8 quantization. Only supported on MLU backend.");

DEFINE_bool(enable_prefix_cache,
            true,
            "Whether to enable the prefix cache for the block manager.");

DEFINE_uint32(xxh3_128bits_seed, 1024, "Default XXH3 128-bits hash seed.");

DEFINE_bool(
    enable_xtensor,
    false,
    "Whether to enable xtensor for model weights with physical page pool.");

DEFINE_int64(
    phy_page_granularity_size,
    2 * 1024 * 1024,
    "Granularity size for one physical page in bytes, default 2MB, when enable "
    "continuous kv cache.");

namespace xllm {

KVCacheConfig KVCacheConfig::from_flags() {
  KVCacheConfig config;
  config.block_size(FLAGS_block_size)
      .max_cache_size(FLAGS_max_cache_size)
      .max_memory_utilization(FLAGS_max_memory_utilization)
      .kv_cache_dtype(FLAGS_kv_cache_dtype)
      .enable_prefix_cache(FLAGS_enable_prefix_cache)
      .xxh3_128bits_seed(FLAGS_xxh3_128bits_seed)
      .enable_xtensor(FLAGS_enable_xtensor)
      .phy_page_granularity_size(FLAGS_phy_page_granularity_size);
  return config;
}

KVCacheConfig& KVCacheConfig::get_instance() {
  static KVCacheConfig config = KVCacheConfig::from_flags();
  return config;
}

void KVCacheConfig::reload_from_flags() {
  KVCacheConfig::get_instance() = KVCacheConfig::from_flags();
}

}  // namespace xllm
