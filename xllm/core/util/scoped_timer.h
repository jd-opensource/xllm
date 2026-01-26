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

#include <glog/logging.h>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

namespace xllm {

class ScopedTimer {
 public:
  explicit ScopedTimer(const std::string& tag, int log_level = 0)
      : tag_(tag),
        log_level_(log_level),
        start_time_(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_);
    double duration_ms = duration.count() / 1000.0;

    std::ostringstream oss;
    oss << tag_ << " took " << std::fixed << std::setprecision(3) << duration_ms
        << " ms";

    if (log_level_ == 0) {
      LOG(INFO) << oss.str();
    } else {
      VLOG(log_level_) << oss.str();
    }
  }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

  double elapsed_ms() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now - start_time_);
    return duration.count() / 1000.0;
  }

 private:
  std::string tag_;
  int log_level_;
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace xllm
