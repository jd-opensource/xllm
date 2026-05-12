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

class DisaggPDConfig final {
 public:
  DisaggPDConfig() = default;
  ~DisaggPDConfig() = default;

  static DisaggPDConfig& get_instance();

  static DisaggPDConfig from_flags();
  static void reload_from_flags();

  PROPERTY(bool, enable_disagg_pd) = false;

  PROPERTY(bool, enable_pd_ooc) = false;

  PROPERTY(int32_t, disagg_pd_port) = 7777;

  PROPERTY(std::string, instance_role) = "DEFAULT";

  PROPERTY(std::string, kv_cache_transfer_type) = "LlmDataDist";

  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  PROPERTY(int32_t, transfer_listen_port) = 26000;
};

}  // namespace xllm
