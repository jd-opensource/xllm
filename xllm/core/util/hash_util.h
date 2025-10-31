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
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "slice.h"

namespace xllm {

constexpr uint32_t MURMUR_HASH3_VALUE_LEN = 16;

struct Murmur3Key {
  uint8_t data[MURMUR_HASH3_VALUE_LEN];

  Murmur3Key() {}
  Murmur3Key(const uint8_t* const input_data) {
    std::memcpy(data, input_data, MURMUR_HASH3_VALUE_LEN);
  }

  bool operator==(const Murmur3Key& other) {
    return std::strncmp(reinterpret_cast<const char*>(data),
                        reinterpret_cast<const char*>(other.data),
                        MURMUR_HASH3_VALUE_LEN);
  }
  std::string debug_string() {
    std::string rt;
    for (int i = 0; i < MURMUR_HASH3_VALUE_LEN; i++) {
      rt += std::to_string(int64_t(data[i])) + " ";
    }
    return rt;
  }
};

struct FixedStringKeyHash {
  size_t operator()(const Murmur3Key& key) const {
    return std::hash<std::string_view>()(std::string_view(
        reinterpret_cast<const char*>(key.data), sizeof(key.data)));
  }
};

struct FixedStringKeyEqual {
  bool operator()(const Murmur3Key& left, const Murmur3Key& right) const {
    return std::strncmp(reinterpret_cast<const char*>(left.data),
                        reinterpret_cast<const char*>(right.data),
                        sizeof(left.data)) == 0;
  }
};

void murmur_hash3(const uint8_t* pre_hash_value,
                  const Slice<int32_t>& token_ids,
                  uint8_t* hash_value);

}  // namespace xllm
