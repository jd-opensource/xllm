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

#include "core/framework/config/distributed_config.h"

#include "core/common/global_flags.h"

DEFINE_string(master_node_addr,
              "127.0.0.1:19888",
              "The master address for multi-node distributed serving(e.g. "
              "10.18.1.1:9999).");

DEFINE_string(
    xtensor_master_node_addr,
    "127.0.0.1:19889",
    "The master address for XTensor distributed service(e.g. 10.18.1.1:9999).");

DEFINE_int32(nnodes, 1, "The number of multi-nodes.");

DEFINE_int32(node_rank, 0, "The node rank.");

DEFINE_string(device_ip, "", "The device IP address for KV cache transfer.");

DEFINE_string(etcd_addr, "", "Etcd adderss for save instance meta info.");

DEFINE_string(etcd_namespace,
              "",
              "Optional etcd namespace prefix for all xllm keys, e.g. prod-a.");

DEFINE_bool(enable_service_routing,
            false,
            "Whether to use xllm service routing.");

DEFINE_double(heart_beat_interval, 0.5, "Heart beat interval.");

DEFINE_int32(etcd_ttl, 3, "Time to live for etcd.");

namespace xllm {

DistributedConfig DistributedConfig::from_flags() {
  DistributedConfig config;
  config.master_node_addr(FLAGS_master_node_addr)
      .xtensor_master_node_addr(FLAGS_xtensor_master_node_addr)
      .nnodes(FLAGS_nnodes)
      .node_rank(FLAGS_node_rank)
      .device_ip(FLAGS_device_ip)
      .etcd_addr(FLAGS_etcd_addr)
      .etcd_namespace(FLAGS_etcd_namespace)
      .enable_service_routing(FLAGS_enable_service_routing)
      .heart_beat_interval(FLAGS_heart_beat_interval)
      .etcd_ttl(FLAGS_etcd_ttl);
  return config;
}

DistributedConfig& DistributedConfig::get_instance() {
  static DistributedConfig config = DistributedConfig::from_flags();
  return config;
}

void DistributedConfig::reload_from_flags() {
  DistributedConfig::get_instance() = DistributedConfig::from_flags();
}

}  // namespace xllm
