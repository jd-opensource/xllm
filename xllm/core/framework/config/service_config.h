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

class ServiceConfig final {
 public:
  ServiceConfig() = default;
  ~ServiceConfig() = default;

  static ServiceConfig& get_instance();

  static ServiceConfig from_flags();
  static void reload_from_flags();

  PROPERTY(std::string, host);

  PROPERTY(int32_t, port) = 8010;

  PROPERTY(int32_t, rpc_idle_timeout_s) = -1;

  PROPERTY(int32_t, rpc_channel_timeout_ms) = -1;

  PROPERTY(int32_t, max_reconnect_count) = 40;

  PROPERTY(int32_t, num_threads) = 8;

  PROPERTY(int32_t, max_concurrent_requests) = 200;

  PROPERTY(int32_t, num_request_handling_threads) = 4;

  PROPERTY(int32_t, num_response_handling_threads) = 4;

  PROPERTY(int32_t, health_check_interval_ms) = 3000;
};

}  // namespace xllm
