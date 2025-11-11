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

#include "env_var.h"

#include <climits>
#include <cstdlib>
#include <cstring>

namespace xllm {
namespace util {

// Environment variable keys
const std::string EXTRA_THREAD_NUM = "EXTRA_THREAD_NUM";

bool get_bool_env(const std::string& key, bool defaultValue) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  std::string strVal(val);
  return (strVal == "1" || strVal == "true" || strVal == "TRUE" ||
          strVal == "True");
}

int get_int_env(const std::string& key, int defaultValue) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  // Use strtol for proper error handling
  char* endptr;
  long int result = std::strtol(val, &endptr, 10);
  // Check if conversion was successful (endptr points to end of string or valid
  // terminator)
  if (endptr == val || *endptr != '\0') {
    return defaultValue;
  }
  // Check for overflow/underflow
  if (result < INT_MIN || result > INT_MAX) {
    return defaultValue;
  }
  return static_cast<int>(result);
}

int get_process_group_test_timeout_seconds() {
  // Default timeout is 4 seconds, but can be overridden via environment
  // variable to accommodate multi-node multi-device communication scenarios
  // where network latency may require a longer timeout period.
  constexpr int kDefaultTimeoutSeconds = 4;
  constexpr const char* kTimeoutEnvVar =
      "XLLM_PROCESS_GROUP_ASYNC_TIMEOUT_SECONDS";
  return get_int_env(kTimeoutEnvVar, kDefaultTimeoutSeconds);
}

}  // namespace util
}  // namespace xllm
