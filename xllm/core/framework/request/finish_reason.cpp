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

#include "finish_reason.h"

#include <glog/logging.h>

namespace xllm {

std::optional<std::string> FinishReason::to_string() {
  switch (value) {
    case Value::NONE:
      return std::nullopt;
    case Value::STOP:
      return "stop";
    case Value::LENGTH:
      return "length";
    case Value::FUNCTION_CALL:
      return "function_call";
    default:
      LOG(WARNING) << "Unknown finish reason: " << static_cast<int>(value);
  }
  return std::nullopt;
}

}  // namespace xllm
