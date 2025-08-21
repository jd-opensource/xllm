#include "utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "partial_json_parser/options.h"
#include "partial_json_parser/parser.h"

namespace xllm {
namespace function_call {

std::string _find_common_prefix(const std::string& s1, const std::string& s2) {
  std::string prefix;
  size_t min_length = std::min(s1.length(), s2.length());

  for (size_t i = 0; i < min_length; ++i) {
    if (s1[i] == s2[i]) {
      prefix += s1[i];
    } else {
      break;
    }
  }

  return prefix;
}

// Convert our Allow enum to partial_json_parser TypeOptions
partial_json_parser::TypeOptions convert_allow_to_type_options(Allow flags) {
  int result = 0;

  if (static_cast<int>(flags) & static_cast<int>(Allow::STR)) {
    result |= partial_json_parser::STR;
  }
  if (static_cast<int>(flags) & static_cast<int>(Allow::NUM)) {
    result |= partial_json_parser::NUM;
  }
  if (static_cast<int>(flags) & static_cast<int>(Allow::ARR)) {
    result |= partial_json_parser::ARR;
  }
  if (static_cast<int>(flags) & static_cast<int>(Allow::OBJ)) {
    result |= partial_json_parser::OBJ;
  }
  if (static_cast<int>(flags) & static_cast<int>(Allow::NULL_TYPE)) {
    result |= partial_json_parser::NULL_TYPE;
  }
  if (static_cast<int>(flags) & static_cast<int>(Allow::BOOL)) {
    result |= partial_json_parser::BOOL;
  }
  if (static_cast<int>(flags) & static_cast<int>(Allow::NAN_TYPE)) {
    result |= partial_json_parser::NAN_TYPE;
  }
  if (static_cast<int>(flags) & static_cast<int>(Allow::INFINITY_TYPE)) {
    result |= partial_json_parser::INFINITY_TYPE;
  }
  if (static_cast<int>(flags) & static_cast<int>(Allow::NEG_INFINITY)) {
    result |= partial_json_parser::NEG_INFINITY;
  }

  return static_cast<partial_json_parser::TypeOptions>(result);
}

std::tuple<nlohmann::json, int> _partial_json_loads(
    const std::string& input_str,
    Allow flags) {
  try {
    // Convert Allow flags to TypeOptions
    auto type_options = convert_allow_to_type_options(flags);

    // Use our C++ partial_json_parser
    std::string completed_json = partial_json_parser::ParseMalformedString(
        input_str, type_options, false);

    // Parse the completed JSON
    nlohmann::json parsed_obj = nlohmann::json::parse(completed_json);

    return std::make_tuple(parsed_obj, static_cast<int>(input_str.length()));

  } catch (const partial_json_parser::MalformedJSONException& e) {
    // Handle malformed JSON - try standard JSON parsing for "Extra data" case
    try {
      nlohmann::json parsed_obj = nlohmann::json::parse(input_str);
      return std::make_tuple(parsed_obj, static_cast<int>(input_str.length()));
    } catch (const nlohmann::json::parse_error& json_e) {
      // If it contains "Extra data", try to parse just the valid part
      std::string error_msg = json_e.what();
      if (error_msg.find("Extra data") != std::string::npos) {
        // Find the position where valid JSON ends
        size_t pos = 0;
        int brace_count = 0;
        bool in_string = false;
        bool escaped = false;

        for (size_t i = 0; i < input_str.length(); ++i) {
          char c = input_str[i];

          if (!in_string) {
            if (c == '{') {
              brace_count++;
            } else if (c == '}') {
              brace_count--;
              if (brace_count == 0) {
                pos = i + 1;
                break;
              }
            } else if (c == '"') {
              in_string = true;
            }
          } else {
            if (escaped) {
              escaped = false;
            } else if (c == '\\') {
              escaped = true;
            } else if (c == '"') {
              in_string = false;
            }
          }
        }

        if (pos > 0) {
          std::string valid_part = input_str.substr(0, pos);
          nlohmann::json parsed_obj = nlohmann::json::parse(valid_part);
          return std::make_tuple(parsed_obj, static_cast<int>(pos));
        }
      }
      throw;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in _partial_json_loads: " << e.what();
    throw;
  }
}

bool _is_complete_json(const std::string& input_str) {
  try {
    nlohmann::json::parse(input_str);
    return true;
  } catch (const nlohmann::json::parse_error&) {
    return false;
  }
}

}  // namespace function_call
}  // namespace xllm