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

#include "core/framework/config/kv_cache_store_config.h"

#include "core/common/global_flags.h"
#include "core/framework/config/config_json_utils.h"

DEFINE_uint32(prefetch_timeout,
              0,
              "Prefetch timeout for prefetch from kv cache store.");

DEFINE_uint32(prefetch_batch_size,
              2,
              "Prefetch from kvcache store copy batch size.");

DEFINE_uint32(layers_wise_copy_batchs, 4, "Layer wise H2D copy batchs.");

DEFINE_double(host_blocks_factor,
              0.0,
              "Host block factor, e.g. host block num = host_blocks_factor * "
              "hbm block num.");

DEFINE_bool(enable_kvcache_store, false, "Whether to use kvcache store.");

DEFINE_bool(enable_cache_upload,
            false,
            "Whether to upload cache info to service. This feature is only "
            "available when service routing is enabled.");

DEFINE_string(store_protocol,
              "tcp",
              "KV cache store protocol(e.g. tcp, rdma).");

DEFINE_string(store_master_server_address,
              "",
              "The address information of the store master service.");

DEFINE_string(store_metadata_server,
              "",
              "The address of the kv cache store metadata service.");

DEFINE_string(store_local_hostname,
              "",
              "The local host name of the kv cache store client.");

DEFINE_bool(enable_control_h2d_block_num,
            false,
            "Whether to control h2d copy block num.");

namespace xllm {

void KVCacheStoreConfig::from_flags() {
  prefetch_timeout(FLAGS_prefetch_timeout)
      .prefetch_batch_size(FLAGS_prefetch_batch_size)
      .layers_wise_copy_batchs(FLAGS_layers_wise_copy_batchs)
      .host_blocks_factor(FLAGS_host_blocks_factor)
      .enable_kvcache_store(FLAGS_enable_kvcache_store)
      .enable_cache_upload(FLAGS_enable_cache_upload)
      .store_protocol(FLAGS_store_protocol)
      .store_master_server_address(FLAGS_store_master_server_address)
      .store_metadata_server(FLAGS_store_metadata_server)
      .store_local_hostname(FLAGS_store_local_hostname)
      .enable_control_h2d_block_num(FLAGS_enable_control_h2d_block_num);
}

void KVCacheStoreConfig::from_json(const JsonReader& json) {
  prefetch_timeout(
      json.value_or<uint32_t>("prefetch_timeout", prefetch_timeout()))
      .prefetch_batch_size(
          json.value_or<uint32_t>("prefetch_batch_size", prefetch_batch_size()))
      .layers_wise_copy_batchs(json.value_or<uint32_t>(
          "layers_wise_copy_batchs", layers_wise_copy_batchs()))
      .host_blocks_factor(
          json.value_or<double>("host_blocks_factor", host_blocks_factor()))
      .enable_kvcache_store(
          json.value_or<bool>("enable_kvcache_store", enable_kvcache_store()))
      .enable_cache_upload(
          json.value_or<bool>("enable_cache_upload", enable_cache_upload()))
      .store_protocol(
          json.value_or<std::string>("store_protocol", store_protocol()))
      .store_master_server_address(json.value_or<std::string>(
          "store_master_server_address", store_master_server_address()))
      .store_metadata_server(json.value_or<std::string>(
          "store_metadata_server", store_metadata_server()))
      .store_local_hostname(json.value_or<std::string>("store_local_hostname",
                                                       store_local_hostname()))
      .enable_control_h2d_block_num(json.value_or<bool>(
          "enable_control_h2d_block_num", enable_control_h2d_block_num()));
}

void KVCacheStoreConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const KVCacheStoreConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, prefetch_timeout);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, prefetch_batch_size);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, layers_wise_copy_batchs);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, host_blocks_factor);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_kvcache_store);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_cache_upload);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_protocol);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_master_server_address);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_metadata_server);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, store_local_hostname);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_control_h2d_block_num);
}

KVCacheStoreConfig& KVCacheStoreConfig::get_instance() {
  static KVCacheStoreConfig config;
  return config;
}

void KVCacheStoreConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
