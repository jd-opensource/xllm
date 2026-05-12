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

#include "core/framework/config/xllm_config.h"

namespace xllm {

XllmConfig XllmConfig::from_flags() {
  XllmConfig config;
  config.service_config(ServiceConfig::from_flags())
      .model_config(ModelConfig::from_flags())
      .load_config(LoadConfig::from_flags())
      .kv_cache_config(KVCacheConfig::from_flags())
      .kv_cache_store_config(KVCacheStoreConfig::from_flags())
      .beam_search_config(BeamSearchConfig::from_flags())
      .scheduler_config(SchedulerConfig::from_flags())
      .parallel_config(ParallelConfig::from_flags())
      .eplb_config(EPLBConfig::from_flags())
      .distributed_config(DistributedConfig::from_flags())
      .disagg_pd_config(DisaggPDConfig::from_flags())
      .speculative_config(SpeculativeConfig::from_flags())
      .profile_config(ProfileConfig::from_flags())
      .execution_config(ExecutionConfig::from_flags())
      .dit_config(DiTConfig::from_flags())
      .rec_config(RecConfig::from_flags());
  return config;
}

XllmConfig& XllmConfig::get_instance() {
  static XllmConfig config = XllmConfig::from_flags();
  return config;
}

void XllmConfig::reload_from_flags() {
  ServiceConfig::reload_from_flags();
  ModelConfig::reload_from_flags();
  LoadConfig::reload_from_flags();
  KVCacheConfig::reload_from_flags();
  KVCacheStoreConfig::reload_from_flags();
  BeamSearchConfig::reload_from_flags();
  SchedulerConfig::reload_from_flags();
  ParallelConfig::reload_from_flags();
  EPLBConfig::reload_from_flags();
  DistributedConfig::reload_from_flags();
  DisaggPDConfig::reload_from_flags();
  SpeculativeConfig::reload_from_flags();
  ProfileConfig::reload_from_flags();
  ExecutionConfig::reload_from_flags();
  DiTConfig::reload_from_flags();
  RecConfig::reload_from_flags();

  XllmConfig config;
  config.service_config(ServiceConfig::get_instance())
      .model_config(ModelConfig::get_instance())
      .load_config(LoadConfig::get_instance())
      .kv_cache_config(KVCacheConfig::get_instance())
      .kv_cache_store_config(KVCacheStoreConfig::get_instance())
      .beam_search_config(BeamSearchConfig::get_instance())
      .scheduler_config(SchedulerConfig::get_instance())
      .parallel_config(ParallelConfig::get_instance())
      .eplb_config(EPLBConfig::get_instance())
      .distributed_config(DistributedConfig::get_instance())
      .disagg_pd_config(DisaggPDConfig::get_instance())
      .speculative_config(SpeculativeConfig::get_instance())
      .profile_config(ProfileConfig::get_instance())
      .execution_config(ExecutionConfig::get_instance())
      .dit_config(DiTConfig::get_instance())
      .rec_config(RecConfig::get_instance());
  XllmConfig::get_instance() = config;
}

}  // namespace xllm
