#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <tuple>

#include "core_types.h"

namespace xllm {
namespace function_call {

// Allow flags for partial JSON parsing
enum class Allow : int {
  STR = 1 << 0,            // 1
  NUM = 1 << 1,            // 2
  ARR = 1 << 2,            // 4
  OBJ = 1 << 3,            // 8
  NULL_TYPE = 1 << 4,      // 16
  BOOL = 1 << 5,           // 32
  NAN_TYPE = 1 << 6,       // 64
  INFINITY_TYPE = 1 << 7,  // 128
  NEG_INFINITY = 1 << 8,   // 256

  // Composite options
  INF = INFINITY_TYPE | NEG_INFINITY,
  SPECIAL = NULL_TYPE | BOOL | INF | NAN_TYPE,
  ATOM = STR | NUM | SPECIAL,
  COLLECTION = ARR | OBJ,
  ALL = ATOM | COLLECTION
};

// Bitwise operations for Allow flags
inline Allow operator|(Allow a, Allow b) {
  return static_cast<Allow>(static_cast<int>(a) | static_cast<int>(b));
}

inline Allow operator&(Allow a, Allow b) {
  return static_cast<Allow>(static_cast<int>(a) & static_cast<int>(b));
}

inline Allow operator~(Allow a) {
  return static_cast<Allow>(~static_cast<int>(a));
}

std::string _find_common_prefix(const std::string& s1, const std::string& s2);

std::tuple<nlohmann::json, int> _partial_json_loads(
    const std::string& input_str,
    Allow flags);

bool _is_complete_json(const std::string& input_str);

}  // namespace function_call
}  // namespace xllm