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

class SpeculativeConfig final {
 public:
  SpeculativeConfig() = default;
  ~SpeculativeConfig() = default;

  static SpeculativeConfig& get_instance();

  static SpeculativeConfig from_flags();
  static void reload_from_flags();

  PROPERTY(std::string, draft_model);

  PROPERTY(std::string, draft_devices) = "npu:0";

  PROPERTY(int32_t, num_speculative_tokens) = 0;

  PROPERTY(std::string, speculative_algorithm) = "MTP";

  PROPERTY(int32_t, speculative_suffix_cache_max_depth) = 64;

  PROPERTY(double, speculative_suffix_max_spec_factor) = 1.0;

  PROPERTY(double, speculative_suffix_max_spec_offset) = 0.0;

  PROPERTY(double, speculative_suffix_min_token_prob) = 0.1;

  PROPERTY(int32_t, speculative_suffix_max_cached_requests) = -1;

  PROPERTY(bool, speculative_suffix_use_tree_spec) = false;

  PROPERTY(bool, enable_opt_validate_probs) = false;

  PROPERTY(bool, enable_atb_spec_kernel) = false;
};

}  // namespace xllm
