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

#include "core/framework/config/disagg_pd_config.h"

#include "core/common/global_flags.h"

DEFINE_bool(enable_disagg_pd,
            false,
            "Whether to enable disaggregated prefill and decode execution.");

DEFINE_bool(
    enable_pd_ooc,
    false,
    "Whether to enable online-offline co-location in disaggregated PD mode.");

DEFINE_int32(disagg_pd_port, 7777, "Port for brpc disagg pd server.");

DEFINE_string(instance_role,
              "DEFAULT",
              "The role of instance(e.g. DEFAULT, PREFILL, DECODE, MIX).");

DEFINE_string(
    kv_cache_transfer_type,
    "LlmDataDist",
    "The type of kv cache transfer(e.g. LlmDataDist, Mooncake, HCCL).");

DEFINE_string(kv_cache_transfer_mode,
              "PUSH",
              "The mode of kv cache transfer(e.g. PUSH, PULL).");

DEFINE_int32(transfer_listen_port, 26000, "The KVCacheTranfer listen port.");

namespace xllm {

DisaggPDConfig DisaggPDConfig::from_flags() {
  DisaggPDConfig config;
  config.enable_disagg_pd(FLAGS_enable_disagg_pd)
      .enable_pd_ooc(FLAGS_enable_pd_ooc)
      .disagg_pd_port(FLAGS_disagg_pd_port)
      .instance_role(FLAGS_instance_role)
      .kv_cache_transfer_type(FLAGS_kv_cache_transfer_type)
      .kv_cache_transfer_mode(FLAGS_kv_cache_transfer_mode)
      .transfer_listen_port(FLAGS_transfer_listen_port);
  return config;
}

DisaggPDConfig& DisaggPDConfig::get_instance() {
  static DisaggPDConfig config = DisaggPDConfig::from_flags();
  return config;
}

void DisaggPDConfig::reload_from_flags() {
  DisaggPDConfig::get_instance() = DisaggPDConfig::from_flags();
}

}  // namespace xllm
