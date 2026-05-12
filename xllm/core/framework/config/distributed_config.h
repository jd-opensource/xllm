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

class DistributedConfig final {
 public:
  DistributedConfig() = default;
  ~DistributedConfig() = default;

  static DistributedConfig& get_instance();

  static DistributedConfig from_flags();
  static void reload_from_flags();

  PROPERTY(std::string, master_node_addr) = "127.0.0.1:19888";

  PROPERTY(std::string, xtensor_master_node_addr) = "127.0.0.1:19889";

  PROPERTY(int32_t, nnodes) = 1;

  PROPERTY(int32_t, node_rank) = 0;

  PROPERTY(std::string, device_ip);

  PROPERTY(std::string, etcd_addr);

  PROPERTY(std::string, etcd_namespace);

  PROPERTY(bool, enable_service_routing) = false;

  PROPERTY(double, heart_beat_interval) = 0.5;

  PROPERTY(int32_t, etcd_ttl) = 3;
};

}  // namespace xllm
