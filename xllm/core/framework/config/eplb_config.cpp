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

#include "core/framework/config/eplb_config.h"

#include "core/common/global_flags.h"

DEFINE_bool(enable_eplb, false, "Whether to use expert parallel load balance.");

DEFINE_int32(redundant_experts_num,
             1,
             "Number of redundant experts on per device.");

DEFINE_int64(eplb_update_interval, 1000, "EPLB update rate.");

DEFINE_double(eplb_update_threshold, 0.8, "EPLB update threshold.");

DEFINE_int32(expert_parallel_degree, 0, "Expert parallel degree.");

DEFINE_string(rank_tablefile, "", "ATB HCCL rank table file.");

namespace xllm {

EPLBConfig EPLBConfig::from_flags() {
  EPLBConfig config;
  config.enable_eplb(FLAGS_enable_eplb)
      .redundant_experts_num(FLAGS_redundant_experts_num)
      .eplb_update_interval(FLAGS_eplb_update_interval)
      .eplb_update_threshold(FLAGS_eplb_update_threshold)
      .expert_parallel_degree(FLAGS_expert_parallel_degree)
      .rank_tablefile(FLAGS_rank_tablefile);
  return config;
}

EPLBConfig& EPLBConfig::get_instance() {
  static EPLBConfig config = EPLBConfig::from_flags();
  return config;
}

void EPLBConfig::reload_from_flags() {
  EPLBConfig::get_instance() = EPLBConfig::from_flags();
}

}  // namespace xllm
