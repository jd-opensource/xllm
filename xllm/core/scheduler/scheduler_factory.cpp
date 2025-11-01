/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "scheduler/scheduler_factory.h"

#include "scheduler/chunked_prefill_scheduler.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/fixsteps_scheduler.h"
#include "scheduler/pd_ooc_scheduler.h"
#include "scheduler/prefill_only_scheduler.h"
#include "scheduler/zero_eviction_scheduler.h"

namespace xllm {

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options) {
  if (options.enable_disagg_pd()) {
    if (options.enable_pd_ooc()) {
      return std::make_unique<PDOOCScheduler>(engine, options);
    } else {
      return std::make_unique<DisaggPDScheduler>(engine, options);
    }
  }

  if (options.enable_chunked_prefill()) {
    if (options.num_speculative_tokens() > 0) {
      return std::make_unique<PrefillOnlyScheduler>(engine, options);
    } else {
      return std::make_unique<ChunkedPrefillScheduler>(engine, options);
    }
  }

  if (FLAGS_use_zero_evict) {
    return std::make_unique<ZeroEvictionScheduler>(engine, options);
  }

  return std::make_unique<ContinuousScheduler>(engine, options);
}

std::unique_ptr<FixStepsScheduler> create_fixsteps_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options) {
  return std::make_unique<FixStepsScheduler>(engine, options);
}

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options) {
  return std::make_unique<DiTDynamicBatchScheduler>(engine, options);
}

}  // namespace xllm
