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

#include <atomic>

namespace xllm {

class RateLimiter final {
 public:
  RateLimiter() = default;

  ~RateLimiter() = default;

  bool is_limited();

  void decrease_one_request();

  void decrease_requests(size_t decrease_requests_num);

 private:
  std::atomic<int32_t> num_concurrent_requests_{0};
};

}  // namespace xllm