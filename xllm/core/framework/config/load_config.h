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

#include "core/common/macros.h"

namespace xllm {

class LoadConfig final {
 public:
  LoadConfig() = default;
  ~LoadConfig() = default;

  static LoadConfig& get_instance();

  static LoadConfig from_flags();
  static void reload_from_flags();

  PROPERTY(bool, enable_manual_loader) = false;

  PROPERTY(bool, enable_rolling_load) = false;

  PROPERTY(int32_t, rolling_load_num_cached_layers) = 2;

  PROPERTY(int32_t, rolling_load_num_rolling_slots) = -1;

  PROPERTY(bool, enable_prefetch_weight) = false;
};

}  // namespace xllm
