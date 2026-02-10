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
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "api_service/call.h"
#include "mm_codec.h"

namespace xllm {
DiTRequest::DiTRequest(const std::string& request_id,
                       const std::string& x_request_id,
                       const std::string& x_request_time,
                       const DiTRequestState& state,
                       const std::string& service_request_id,
                       const std::string& source_xservice_addr)
    : RequestBase(request_id,
                  x_request_id,
                  x_request_time,
                  service_request_id,
                  source_xservice_addr),
      state_(state) {}

bool DiTRequest::finished() const { return true; }

void DiTRequest::log_statistic(double total_latency) {
  LOG(INFO) << "x-request-id: " << x_request_id_ << ", "
            << "x-request-time: " << x_request_time_ << ", "
            << "request_id: " << request_id_ << ", "
            << "total_latency: " << total_latency * 1000 << "ms";
}

void DiTRequest::handle_forward_output(torch::Tensor output) {
  int count = state_.generation_params().num_images_per_prompt;
  output_.tensors = torch::chunk(output, count);
}

const DiTRequestOutput DiTRequest::generate_output() {
  DiTRequestOutput output;
  output.request_id = request_id_;
  output.service_request_id = service_request_id_;
  output.status = Status(StatusCode::OK);
  output.finished = finished();
  output.cancelled = false;

  DiTGenerationOutput result;
  result.height = state_.generation_params().height;
  result.width = state_.generation_params().width;
  result.seed = state_.generation_params().seed;

  OpenCVImageEncoder encoder;
  int count = state_.generation_params().num_images_per_prompt;
  for (size_t idx = 0; idx < count; ++idx) {
    torch::Tensor image =
        output_.tensors[idx].squeeze(0).cpu().to(torch::kFloat32).contiguous();
    encoder.encode(image, result.image);
    output.outputs.push_back(result);
  }

  return output;
}

}  // namespace xllm
