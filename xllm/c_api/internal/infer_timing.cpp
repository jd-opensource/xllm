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

#include "c_api/internal/infer_timing.h"

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace xllm {
namespace c_api_infer_timing {
namespace {

std::mutex g_mutex;
std::unordered_map<std::string, XLLM_InferTimingDetail> g_details;

void init_detail(XLLM_InferTimingDetail* detail,
                 const std::string& request_id) {
  std::memset(detail, 0, sizeof(*detail));
  std::strncpy(
      detail->request_id, request_id.c_str(), sizeof(detail->request_id) - 1);
  detail->request_id[sizeof(detail->request_id) - 1] = '\0';
}

XLLM_InferTimingDetail* find_detail(const std::string& request_id) {
  const auto it = g_details.find(request_id);
  if (it == g_details.end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace

void begin_request(const std::string& request_id) {
  if (request_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  init_detail(&g_details[request_id], request_id);
}

void ensure_request(const std::string& request_id) {
  if (request_id.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_details.find(request_id) == g_details.end()) {
    init_detail(&g_details[request_id], request_id);
  }
}

void set_convert_input_tensors_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (auto* detail = find_detail(request_id)) {
    detail->convert_input_tensors_us = us;
  }
}

void set_threadpool_wait_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (auto* detail = find_detail(request_id)) {
    detail->threadpool_wait_us = us;
  }
}

void set_build_request_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (auto* detail = find_detail(request_id)) {
    detail->build_request_us = us;
  }
}

void set_scheduler_infer_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (auto* detail = find_detail(request_id)) {
    detail->scheduler_infer_us = us;
  }
}

void set_generate_output_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (auto* detail = find_detail(request_id)) {
    detail->generate_output_us = us;
  }
}

void set_build_response_us(const std::string& request_id, int64_t us) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (auto* detail = find_detail(request_id)) {
    detail->build_response_us = us;
  }
}

bool take(const std::string& request_id, XLLM_InferTimingDetail* out) {
  if (out == nullptr || request_id.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(g_mutex);
  const auto it = g_details.find(request_id);
  if (it == g_details.end()) {
    return false;
  }
  *out = it->second;
  g_details.erase(it);
  return true;
}

}  // namespace c_api_infer_timing
}  // namespace xllm
