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

#pragma once

#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "scheduler/dit_scheduler.h"
#include "scheduler/fixsteps_scheduler.h"

namespace xllm {

std::unique_ptr<ContinuousScheduler> create_continuous_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options);

std::unique_ptr<FixStepsScheduler> create_fixsteps_scheduler(
    Engine* engine,
    ContinuousScheduler::Options options);

std::unique_ptr<DiTScheduler> create_dit_scheduler(
    DiTEngine* engine,
    DiTScheduler::Options options);

}  // namespace xllm
