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
#include <limits>
#include <nlohmann/json_fwd.hpp>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;

class ProfileConfig final {
 public:
  ProfileConfig() = default;
  ~ProfileConfig() = default;

  static ProfileConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "PROFILE OPTIONS",
        {"enable_profile_step_time",
         "enable_profile_token_budget",
         "enable_latency_aware_schedule",
         "profile_max_prompt_length",
         "max_global_ttft_ms",
         "max_global_tpot_ms",
         "enable_profile_kv_blocks",
         "disable_ttft_profiling",
         "enable_forward_interruption"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_profile_step_time) = false;

  PROPERTY(bool, enable_profile_token_budget) = false;

  PROPERTY(bool, enable_latency_aware_schedule) = false;

  PROPERTY(int32_t, profile_max_prompt_length) = 2048;

  PROPERTY(int32_t, max_global_ttft_ms) = std::numeric_limits<int32_t>::max();

  PROPERTY(int32_t, max_global_tpot_ms) = std::numeric_limits<int32_t>::max();

  PROPERTY(bool, enable_profile_kv_blocks) = true;

  PROPERTY(bool, disable_ttft_profiling) = false;

  PROPERTY(bool, enable_forward_interruption) = false;
};

}  // namespace xllm
