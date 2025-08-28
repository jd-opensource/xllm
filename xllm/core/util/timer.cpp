/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "timer.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>

namespace xllm {

Timer::Timer() : start_(absl::Now()) {}

// reset the timer
void Timer::reset() { start_ = absl::Now(); }

// get the elapsed time in seconds
double Timer::elapsed_seconds() const {
  return absl::ToDoubleSeconds(absl::Now() - start_);
}

}  // namespace xllm