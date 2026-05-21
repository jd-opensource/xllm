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
#include "core/framework/config/config_json_utils.h"

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

void DisaggPDConfig::from_flags() {
  enable_disagg_pd(FLAGS_enable_disagg_pd)
      .enable_pd_ooc(FLAGS_enable_pd_ooc)
      .disagg_pd_port(FLAGS_disagg_pd_port)
      .instance_role(FLAGS_instance_role)
      .kv_cache_transfer_type(FLAGS_kv_cache_transfer_type)
      .kv_cache_transfer_mode(FLAGS_kv_cache_transfer_mode)
      .transfer_listen_port(FLAGS_transfer_listen_port);
}

void DisaggPDConfig::from_json(const JsonReader& json) {
  enable_disagg_pd(json.value_or<bool>("enable_disagg_pd", enable_disagg_pd()))
      .enable_pd_ooc(json.value_or<bool>("enable_pd_ooc", enable_pd_ooc()))
      .disagg_pd_port(
          json.value_or<int32_t>("disagg_pd_port", disagg_pd_port()))
      .instance_role(
          json.value_or<std::string>("instance_role", instance_role()))
      .kv_cache_transfer_type(json.value_or<std::string>(
          "kv_cache_transfer_type", kv_cache_transfer_type()))
      .kv_cache_transfer_mode(json.value_or<std::string>(
          "kv_cache_transfer_mode", kv_cache_transfer_mode()))
      .transfer_listen_port(json.value_or<int32_t>("transfer_listen_port",
                                                   transfer_listen_port()));
}

DisaggPDConfig& DisaggPDConfig::get_instance() {
  static DisaggPDConfig config;
  return config;
}

void DisaggPDConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
