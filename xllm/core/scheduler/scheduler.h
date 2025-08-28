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

#pragma once

#include <absl/time/time.h>

#include <memory>
#include <string>

#include "framework/request/request.h"

namespace xllm {

class Scheduler {
 public:
  virtual ~Scheduler() = default;

  // add a new request to scheduler.
  virtual bool add_request(std::shared_ptr<Request>& request) = 0;

  // scheduler forward execute
  virtual void step(const absl::Duration& timeout) = 0;

  // offline running
  virtual void generate() = 0;

  // incr/decr pending requests
  virtual void incr_pending_requests(size_t count) {}
  virtual void decr_pending_requests() {}
  virtual size_t num_pending_requests() { return 0; }

  virtual uint32_t get_waiting_requests_num() const = 0;
};

}  // namespace xllm
