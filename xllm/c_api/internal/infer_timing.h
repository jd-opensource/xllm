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

#include <cstdint>
#include <string>

#include "c_api/types.h"

namespace xllm {
namespace c_api_infer_timing {

// Create or reset timing state for a request (keyed by request_id).
void begin_request(const std::string& request_id);

// Create timing state only when it does not exist yet.
void ensure_request(const std::string& request_id);

void set_convert_input_tensors_us(const std::string& request_id, int64_t us);
void set_threadpool_wait_us(const std::string& request_id, int64_t us);
void set_build_request_us(const std::string& request_id, int64_t us);
void set_scheduler_infer_us(const std::string& request_id, int64_t us);
void set_generate_output_us(const std::string& request_id, int64_t us);
void set_build_response_us(const std::string& request_id, int64_t us);

// Move timing out of the store for the given request_id.
bool take(const std::string& request_id, XLLM_InferTimingDetail* out);

}  // namespace c_api_infer_timing
}  // namespace xllm
