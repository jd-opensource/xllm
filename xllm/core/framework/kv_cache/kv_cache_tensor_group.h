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

#include <cstdint>
#include <map>
#include <string>

#include "kv_cache_tensor_role.h"

namespace xllm {

class PrefixCacheGroup {
 public:
  enum Value : int8_t {
    C1 = 0,
    SINGLE = 1,
    C4 = 2,
    C128 = 3,
    INVALID = -1,
  };

  constexpr PrefixCacheGroup(Value v) : value_(v) {}
  PrefixCacheGroup(const std::string& str) {
    if (str == "C1" || str == "c1") {
      value_ = C1;
    } else if (str == "SINGLE" || str == "single") {
      value_ = SINGLE;
    } else if (str == "C4" || str == "c4") {
      value_ = C4;
    } else if (str == "C128" || str == "c128") {
      value_ = C128;
    } else {
      value_ = INVALID;
    }
  }

  PrefixCacheGroup() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  constexpr bool operator==(PrefixCacheGroup rhs) const {
    return value_ == rhs.value_;
  }
  constexpr bool operator!=(PrefixCacheGroup rhs) const {
    return value_ != rhs.value_;
  }
  constexpr bool operator==(Value rhs) const { return value_ == rhs; }
  constexpr bool operator!=(Value rhs) const { return value_ != rhs; }

  constexpr const char* to_string() const {
    if (value_ == C1) {
      return "c1";
    } else if (value_ == SINGLE) {
      return "single";
    } else if (value_ == C4) {
      return "c4";
    } else if (value_ == C128) {
      return "c128";
    } else {
      return "invalid";
    }
  }

 private:
  Value value_;
};

using KVCacheTensorGroup = PrefixCacheGroup;

}  // namespace xllm
