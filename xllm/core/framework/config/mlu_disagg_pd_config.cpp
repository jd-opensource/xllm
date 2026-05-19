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

#include "core/framework/config/mlu_disagg_pd_config.h"

#include <glog/logging.h>

#include <string>

namespace xllm {
namespace {

bool supports_prefix_cache(const std::string& instance_role) {
  return instance_role == "PREFILL" || instance_role == "MIX";
}

}  // namespace

void normalize_mlu_disagg_pd_config(DisaggPDConfig& disagg_pd_config,
                                    KVCacheConfig& kv_cache_config,
                                    SchedulerConfig& scheduler_config) {
  if (disagg_pd_config.kv_cache_transfer_type() != "Mooncake") {
    LOG(WARNING) << "MLU disaggregated PD requires "
                 << "kv_cache_transfer_type=Mooncake; forcing from "
                 << disagg_pd_config.kv_cache_transfer_type()
                 << " to Mooncake.";
    disagg_pd_config.kv_cache_transfer_type("Mooncake");
  }
  if (disagg_pd_config.kv_cache_transfer_mode() != "PUSH") {
    LOG(WARNING) << "MLU disaggregated PD requires "
                 << "kv_cache_transfer_mode=PUSH; forcing from "
                 << disagg_pd_config.kv_cache_transfer_mode() << " to PUSH.";
    disagg_pd_config.kv_cache_transfer_mode("PUSH");
  }
  if (kv_cache_config.kv_cache_dtype() != "auto") {
    LOG(WARNING) << "MLU disaggregated PD requires kv_cache_dtype=auto; "
                 << "forcing from " << kv_cache_config.kv_cache_dtype()
                 << " to auto.";
    kv_cache_config.kv_cache_dtype("auto");
  }
  if (scheduler_config.enable_schedule_overlap()) {
    LOG(WARNING) << "MLU disaggregated PD does not support schedule overlap; "
                 << "forcing enable_schedule_overlap=false.";
    scheduler_config.enable_schedule_overlap(false);
  }
  if (kv_cache_config.enable_prefix_cache() &&
      !supports_prefix_cache(disagg_pd_config.instance_role())) {
    LOG(WARNING) << "MLU disaggregated PD role "
                 << disagg_pd_config.instance_role()
                 << " does not support prefix cache; "
                 << "forcing enable_prefix_cache=false.";
    kv_cache_config.enable_prefix_cache(false);
  }
  if (disagg_pd_config.enable_pd_ooc()) {
    LOG(WARNING) << "MLU disaggregated PD does not support pd_ooc; "
                 << "forcing enable_pd_ooc=false.";
    disagg_pd_config.enable_pd_ooc(false);
  }
}

}  // namespace xllm
