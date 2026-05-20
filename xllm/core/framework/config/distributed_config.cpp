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
#include "core/framework/config/config_json_utils.h"

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

void DistributedConfig::from_flags() {
  master_node_addr(FLAGS_master_node_addr)
      .xtensor_master_node_addr(FLAGS_xtensor_master_node_addr)
      .nnodes(FLAGS_nnodes)
      .node_rank(FLAGS_node_rank)
      .device_ip(FLAGS_device_ip)
      .etcd_addr(FLAGS_etcd_addr)
      .etcd_namespace(FLAGS_etcd_namespace)
      .enable_service_routing(FLAGS_enable_service_routing)
      .heart_beat_interval(FLAGS_heart_beat_interval)
      .etcd_ttl(FLAGS_etcd_ttl);
}

void DistributedConfig::from_json(const JsonReader& json) {
  master_node_addr(
      json.value_or<std::string>("master_node_addr", master_node_addr()))
      .xtensor_master_node_addr(json.value_or<std::string>(
          "xtensor_master_node_addr", xtensor_master_node_addr()))
      .nnodes(json.value_or<int32_t>("nnodes", nnodes()))
      .node_rank(json.value_or<int32_t>("node_rank", node_rank()))
      .device_ip(json.value_or<std::string>("device_ip", device_ip()))
      .etcd_addr(json.value_or<std::string>("etcd_addr", etcd_addr()))
      .etcd_namespace(
          json.value_or<std::string>("etcd_namespace", etcd_namespace()))
      .enable_service_routing(json.value_or<bool>("enable_service_routing",
                                                  enable_service_routing()))
      .heart_beat_interval(
          json.value_or<double>("heart_beat_interval", heart_beat_interval()))
      .etcd_ttl(json.value_or<int32_t>("etcd_ttl", etcd_ttl()));
}

DistributedConfig& DistributedConfig::get_instance() {
  static DistributedConfig config;
  return config;
}

void DistributedConfig::initialize() {
  from_flags();
  if (!FLAGS_config_json_file.empty()) {
    from_json(config::load_json_file(FLAGS_config_json_file));
  }
}

}  // namespace xllm
