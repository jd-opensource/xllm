/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <type_traits>

namespace xllm {
namespace util {

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T> ceil_div(T value, T divisor) {
  CHECK_GT(divisor, 0) << "divisor must be positive.";
  return value / divisor + static_cast<T>(value % divisor != 0);
}

}  // namespace util
}  // namespace xllm
