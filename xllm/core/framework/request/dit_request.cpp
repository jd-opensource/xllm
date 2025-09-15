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

#include "dit_request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <vector>

#include "api_service/call.h"

namespace xllm {
DiTRequest::DiTRequest(const std::string& request_id,
                       const std::string& x_request_id,
                       const std::string& x_request_time,
                       const DiTRequestState& state,
                       const std::string& service_request_id,
                       bool offline,
                       int32_t slo_ms,
                       RequestPriority priority)
    : created_time_(absl::Now()),
      request_id_(request_id),
      service_request_id_(service_request_id),
      x_request_id_(x_request_id),
      x_request_time_(x_request_time),
      state_(state),
      offline_(offline),
      slo_ms_(slo_ms),
      priority_(priority) {}

bool DiTRequest::finished() const { return true; }

void DiTRequest::log_statistic(double total_latency) {
  LOG(INFO) << "x-request-id: " << x_request_id_ << ", "
            << "x-request-time: " << x_request_time_ << ", "
            << "request_id: " << request_id_ << ", "
            << "total_latency: " << total_latency * 1000 << "ms";
}

DiTRequestOutput DiTRequest::generate_dit_output(DiTForwardOutput dit_output) {
  DiTRequestOutput output;
  output.request_id = request_id_;
  output.service_request_id = service_request_id_;
  output.status = Status(StatusCode::OK);
  output.finished = finished();
  output.cancelled = false;
  DiTGenerationOutput result;
  result.image_tensor = dit_output.image;
  output.outputs.push_back(result);
  return output;
}
}  // namespace xllm