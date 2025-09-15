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
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "dit_request_output.h"
#include "dit_request_state.h"
#include "request.h"
#include "runtime/dit_forward_params.h"

namespace xllm {

class DiTRequest {
 public:
  DiTRequest(const std::string& request_id,
             const std::string& x_request_id,
             const std::string& x_request_time,
             const DiTRequestState& state,
             const std::string& service_request_id = "",
             bool offline = false,
             int32_t slo_ms = 0,
             RequestPriority priority = RequestPriority::NORMAL);

  bool finished() const;

  DiTRequestOutput generate_dit_output(DiTForwardOutput dit_output);

  void log_statistic(double total_latency);

  const std::string& request_id() const { return request_id_; }

  const std::string& service_request_id() const { return service_request_id_; }

  const std::string& x_request_id() const { return x_request_id_; }

  const std::string& x_request_time() const { return x_request_time_; }

  const bool offline() const { return offline_; }
  const int32_t slo_ms() const { return slo_ms_; }
  const RequestPriority priority() const { return priority_; }

  DiTRequestState& state() { return state_; }

 private:
  // request create time
  absl::Time created_time_;

  std::string request_id_;

  std::string service_request_id_;

  // x-request-id header value from client
  std::string x_request_id_;

  // x-request-time header value from client
  std::string x_request_time_;

  DiTRequestState state_;

  std::atomic<bool> cancelled_{false};

  bool offline_;

  int32_t slo_ms_;

  RequestPriority priority_;
};

}  // namespace xllm