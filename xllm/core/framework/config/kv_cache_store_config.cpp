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

DEFINE_uint32(prefetch_timeout,
              0,
              "Prefetch timeout for prefetch from kv cache store.");

DEFINE_uint32(prefetch_bacth_size,
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

KVCacheStoreConfig KVCacheStoreConfig::from_flags() {
  KVCacheStoreConfig config;
  config.prefetch_timeout(FLAGS_prefetch_timeout)
      .prefetch_bacth_size(FLAGS_prefetch_bacth_size)
      .layers_wise_copy_batchs(FLAGS_layers_wise_copy_batchs)
      .host_blocks_factor(FLAGS_host_blocks_factor)
      .enable_kvcache_store(FLAGS_enable_kvcache_store)
      .enable_cache_upload(FLAGS_enable_cache_upload)
      .store_protocol(FLAGS_store_protocol)
      .store_master_server_address(FLAGS_store_master_server_address)
      .store_metadata_server(FLAGS_store_metadata_server)
      .store_local_hostname(FLAGS_store_local_hostname)
      .enable_control_h2d_block_num(FLAGS_enable_control_h2d_block_num);
  return config;
}

KVCacheStoreConfig& KVCacheStoreConfig::get_instance() {
  static KVCacheStoreConfig config = KVCacheStoreConfig::from_flags();
  return config;
}

void KVCacheStoreConfig::reload_from_flags() {
  KVCacheStoreConfig::get_instance() = KVCacheStoreConfig::from_flags();
}

}  // namespace xllm
