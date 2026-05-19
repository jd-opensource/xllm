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

#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"

namespace xllm {

void normalize_mlu_disagg_pd_config(DisaggPDConfig& disagg_pd_config,
                                    KVCacheConfig& kv_cache_config,
                                    SchedulerConfig& scheduler_config);

}  // namespace xllm
