/* Copyright 2026 The xLLM Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include <ostream>

namespace xllm {
class DebugLogStream {
 public:
  DebugLogStream() : is_enabled_(VLOG_IS_ON(1)) {
    if (is_enabled_) {
      stream_ << "[DEBUG] ";
    }
  }

  ~DebugLogStream() {
    if (is_enabled_) {
      VLOG(1) << stream_.str();
    }
  }

  template <typename T>
  DebugLogStream& operator<<(T&& value) {
    if (is_enabled_) {
      stream_ << std::forward<T>(value);
    }
    return *this;
  }

  DebugLogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
    if (is_enabled_) {
      stream_ << manip;
    }
    return *this;
  }

 private:
  bool is_enabled_;
  std::ostringstream stream_;
};

#define LOG_DEBUG() \
  if (!VLOG_IS_ON(1)) (void)0; else DebugLogStream()

// use like this:
// LOG_DEBUG() << "test x-request-id: " << x_request_id_;
// curl http://localhost:port/logging  to see the log level
// curl http://localhost:port/logging -d "level=DEBUG"  to open DEBUG log level
// curl http://localhost:port/logging -d "level=INFO"  to close DEBUG log level
}  // namespace xllm