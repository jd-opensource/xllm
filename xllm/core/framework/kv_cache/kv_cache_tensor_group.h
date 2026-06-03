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
#include <unordered_set>

#include "kv_cache_tensor_role.h"

namespace xllm {

class KVCacheTensorGroup {
 public:
  enum Value : int8_t {
    C1 = 0,
    SINGLE = 1,
    INVALID = -1,
  };

  constexpr KVCacheTensorGroup(Value v) : value_(v) {}
  KVCacheTensorGroup(const std::string& str) {
    if (str == "C1" || str == "c1") {
      value_ = C1;
    } else if (str == "SINGLE" || str == "single") {
      value_ = SINGLE;
    } else {
      value_ = INVALID;
    }
  }

  KVCacheTensorGroup() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  constexpr bool operator==(KVCacheTensorGroup rhs) const {
    return value_ == rhs.value_;
  }
  constexpr bool operator!=(KVCacheTensorGroup rhs) const {
    return value_ != rhs.value_;
  }
  constexpr bool operator==(Value rhs) const { return value_ == rhs; }
  constexpr bool operator!=(Value rhs) const { return value_ != rhs; }

  bool contains(KVCacheTensorRole role) const {
    static const std::map<Value, std::unordered_set<KVCacheTensorRole::Value>>
        kGroupRoles = {
            {C1,
             {KVCacheTensorRole::KEY,
              KVCacheTensorRole::VALUE,
              KVCacheTensorRole::INDEX}},
            {SINGLE,
             {KVCacheTensorRole::CONV,
              KVCacheTensorRole::SSM,
              KVCacheTensorRole::SWA}},
        };

    const auto group_it = kGroupRoles.find(value_);
    if (group_it == kGroupRoles.end()) {
      return false;
    }

    return group_it->second.find(static_cast<KVCacheTensorRole::Value>(role)) !=
           group_it->second.end();
  }

  constexpr const char* to_string() const {
    if (value_ == C1) {
      return "c1";
    } else if (value_ == SINGLE) {
      return "single";
    } else {
      return "invalid";
    }
  }

 private:
  Value value_;
};

}  // namespace xllm
