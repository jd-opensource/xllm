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

#include "rate_limiter.h"

#include <gflags/gflags.h>

#include "common/global_flags.h"
#include "common/metrics.h"

namespace xllm {

bool RateLimiter::is_limited() {
  if (FLAGS_max_concurrent_requests > 0) {
    int32_t num_requests =
        num_concurrent_requests_.load(std::memory_order_relaxed);
    if (num_requests >= FLAGS_max_concurrent_requests) {
      COUNTER_INC(server_request_total_limit);
      return true;
    }
  }
  num_concurrent_requests_.fetch_add(1, std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));

  return false;
}

void RateLimiter::decrease_one_request() {
  num_concurrent_requests_.fetch_sub(1, std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));
}

void RateLimiter::decrease_requests(size_t decrease_requests_num) {
  num_concurrent_requests_.fetch_sub(decrease_requests_num,
                                     std::memory_order_relaxed);
  GAUGE_SET(num_concurrent_requests,
            num_concurrent_requests_.load(std::memory_order_relaxed));
}

}  // namespace xllm
