#ifndef PARTIAL_JSON_PARSER_PARSER_H
#define PARTIAL_JSON_PARSER_PARSER_H

#include <stdexcept>
#include <string>

#include "options.h"

namespace partial_json_parser {

// Exception class for malformed JSON - matches Go error handling
class MalformedJSONException : public std::runtime_error {
 public:
  explicit MalformedJSONException(const std::string& message)
      : std::runtime_error(message) {}
};

// Structure that matches the Go jsonCompletion struct exactly
struct JsonCompletion {
  int index;
  std::string string;

  JsonCompletion(int idx = 0, const std::string& str = "")
      : index(idx), string(str) {}
};

// Main API function - matches Go ParseMalformedString exactly
std::string ParseMalformedString(const std::string& malformed,
                                 TypeOptions options,
                                 bool format = false);

// Internal parsing functions - matching Go implementation structure
std::string parseJson(const std::string& jsonString, TypeOptions allowed);
JsonCompletion completeAny(const std::string& jsonString,
                           TypeOptions allowed,
                           bool topLevel);
JsonCompletion completeString(const std::string& jsonString,
                              TypeOptions allowed);
JsonCompletion completeArray(const std::string& jsonString,
                             TypeOptions allowed);
JsonCompletion completeObject(const std::string& jsonString,
                              TypeOptions allowed);
JsonCompletion completeNumber(const std::string& jsonString,
                              TypeOptions allowed,
                              bool topLevel);

// Utility functions - matching Go implementation
int skipBlank(const std::string& text, int index);
std::string formatJson(const std::string& jsonString);

}  // namespace partial_json_parser

#endif  // PARTIAL_JSON_PARSER_PARSER_H