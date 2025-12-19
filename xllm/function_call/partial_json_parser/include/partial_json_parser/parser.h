#ifndef PARTIAL_JSON_PARSER_PARSER_H
#define PARTIAL_JSON_PARSER_PARSER_H

#include <stdexcept>
#include <string>

#include "options.h"

namespace partial_json_parser {

class MalformedJSONException : public std::runtime_error {
 public:
  explicit MalformedJSONException(const std::string& message)
      : std::runtime_error(message) {}
};

struct JsonCompletion {
  int32_t index;
  std::string string;

  JsonCompletion(int32_t idx = 0, const std::string& str = "")
      : index(idx), string(str) {}
};

std::string parse_malformed_string(const std::string& malformed,
                                   TypeOptions options,
                                   bool format = false);

std::string parse_json(const std::string& json_string, TypeOptions allowed);
JsonCompletion complete_any(const std::string& json_string,
                            TypeOptions allowed,
                            bool top_level);
JsonCompletion complete_string(const std::string& json_string,
                               TypeOptions allowed);
JsonCompletion complete_array(const std::string& json_string,
                              TypeOptions allowed);
JsonCompletion complete_object(const std::string& json_string,
                               TypeOptions allowed);
JsonCompletion complete_number(const std::string& json_string,
                               TypeOptions allowed,
                               bool top_level);

int32_t skip_blank(const std::string& text, int32_t index);
std::string format_json(const std::string& json_string);

}  // namespace partial_json_parser

#endif  // PARTIAL_JSON_PARSER_PARSER_H