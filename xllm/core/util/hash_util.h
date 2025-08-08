#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "util/slice.h"

namespace xllm {

constexpr uint32_t MURMUR_HASH3_VALUE_LEN = 16;

struct Murmur3Key {
  uint8_t data[MURMUR_HASH3_VALUE_LEN];

  Murmur3Key() {}
  Murmur3Key(const uint8_t* const input_data) {
    memcpy(data, input_data, MURMUR_HASH3_VALUE_LEN);
  }

  bool operator==(const Murmur3Key& other) {
    return strncmp(reinterpret_cast<const char*>(data),
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

  static uint32_t get_key_len() { return MURMUR_HASH3_VALUE_LEN; }
};

struct FixedStringKeyHash {
  size_t operator()(const Murmur3Key& key) const {
    return std::hash<std::string_view>()(std::string_view(
        reinterpret_cast<const char*>(key.data), sizeof(key.data)));
  }
};

struct FixedStringKeyEqual {
  bool operator()(const Murmur3Key& left, const Murmur3Key& right) const {
    return strncmp(reinterpret_cast<const char*>(left.data),
                   reinterpret_cast<const char*>(right.data),
                   sizeof(left.data)) == 0;
  }
};

void print_hex_array(uint8_t* array, uint32_t len);

class MurMurHash3 {
 public:
  uint32_t get_hash_value_len() { return MURMUR_HASH3_VALUE_LEN; }

  void hash(const uint8_t* pre_hash_value,
            const Slice<int32_t>& token_ids,
            uint8_t* hash_value);
};

}  // namespace xllm
