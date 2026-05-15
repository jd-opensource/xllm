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

#include <string>

#include "core/common/macros.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/profile_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"

namespace xllm {

class XllmConfig final {
 public:
  XllmConfig() = default;
  ~XllmConfig() = default;

  static XllmConfig& get_instance();

  static XllmConfig from_flags();
  static void reload_from_flags();
  static void reload_from_configs();

  PROPERTY(ServiceConfig, service_config);

  PROPERTY(ModelConfig, model_config);

  PROPERTY(LoadConfig, load_config);

  PROPERTY(KVCacheConfig, kv_cache_config);

  PROPERTY(KVCacheStoreConfig, kv_cache_store_config);

  PROPERTY(BeamSearchConfig, beam_search_config);

  PROPERTY(SchedulerConfig, scheduler_config);

  PROPERTY(ParallelConfig, parallel_config);

  PROPERTY(EPLBConfig, eplb_config);

  PROPERTY(DistributedConfig, distributed_config);

  PROPERTY(DisaggPDConfig, disagg_pd_config);

  PROPERTY(SpeculativeConfig, speculative_config);

  PROPERTY(ProfileConfig, profile_config);

  PROPERTY(ExecutionConfig, execution_config);

  PROPERTY(DiTConfig, dit_config);

  PROPERTY(RecConfig, rec_config);
};

}  // namespace xllm
