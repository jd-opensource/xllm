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

#include "core/framework/config/profile_config.h"

#include <limits>

#include "core/common/global_flags.h"

DEFINE_bool(enable_profile_step_time,
            false,
            "Whether to enable profile step time.");

DEFINE_bool(enable_profile_token_budget,
            false,
            "Whether to enable profile token budget.");

DEFINE_bool(enable_latency_aware_schedule,
            false,
            "use predicted latency for latency aware schedule.");

DEFINE_int32(profile_max_prompt_length,
             2048,
             "The max prompt length for profile.");

DEFINE_int32(max_global_ttft_ms,
             std::numeric_limits<int32_t>::max(),
             "all requests use single global ttft");

DEFINE_int32(max_global_tpot_ms,
             std::numeric_limits<int32_t>::max(),
             "all requests use single global ttft");

DEFINE_bool(enable_profile_kv_blocks,
            true,
            "true if generate kv cache for profile");

DEFINE_bool(disable_ttft_profiling,
            false,
            "Whether to disable TTFT profiling.");

DEFINE_bool(enable_forward_interruption,
            false,
            "Whether to enable forward interruption.");

namespace xllm {

ProfileConfig ProfileConfig::from_flags() {
  ProfileConfig config;
  config.enable_profile_step_time(FLAGS_enable_profile_step_time)
      .enable_profile_token_budget(FLAGS_enable_profile_token_budget)
      .enable_latency_aware_schedule(FLAGS_enable_latency_aware_schedule)
      .profile_max_prompt_length(FLAGS_profile_max_prompt_length)
      .max_global_ttft_ms(FLAGS_max_global_ttft_ms)
      .max_global_tpot_ms(FLAGS_max_global_tpot_ms)
      .enable_profile_kv_blocks(FLAGS_enable_profile_kv_blocks)
      .disable_ttft_profiling(FLAGS_disable_ttft_profiling)
      .enable_forward_interruption(FLAGS_enable_forward_interruption);
  return config;
}

ProfileConfig& ProfileConfig::get_instance() {
  static ProfileConfig config = ProfileConfig::from_flags();
  return config;
}

void ProfileConfig::reload_from_flags() {
  ProfileConfig::get_instance() = ProfileConfig::from_flags();
}

}  // namespace xllm
