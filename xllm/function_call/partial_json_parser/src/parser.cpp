/**
 * Partial JSON Parser - C++ Implementation
 *
 * Based on:
 * - Python: https://github.com/promplate/partial-json-parser
 * - Go: https://github.com/blaze2305/partial-json-parser
 *
 * Thanks to the original authors for their excellent work.
 */

#include "partial_json_parser/parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <nlohmann/json.hpp>
#include <sstream>

namespace partial_json_parser {

std::string formatJson(const std::string& jsonString) {
  try {
    auto json = nlohmann::json::parse(jsonString);
    return json.dump(1);
  } catch (const std::exception& e) {
    return jsonString;
  }
}

// Main API function
std::string ParseMalformedString(const std::string& malformed,
                                 TypeOptions options,
                                 bool format) {
  std::string str = malformed;

  // Trim whitespace
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(), [](unsigned char ch) {
              return !std::isspace(ch);
            }));
  str.erase(std::find_if(str.rbegin(),
                         str.rend(),
                         [](unsigned char ch) { return !std::isspace(ch); })
                .base(),
            str.end());

  if (str.empty()) {
    throw MalformedJSONException("string is empty; cannot parse");
  }

  std::string jsonString = parseJson(malformed, options);

  if (format) {
    return formatJson(jsonString);
  } else {
    return jsonString;
  }
}

int skipBlank(const std::string& text, int index) {
  int i = index;
  while (
      i < static_cast<int>(text.length()) &&
      (std::isspace(static_cast<unsigned char>(text[i])) || text[i] == '\n')) {
    i += 1;
  }
  return i;
}

std::string parseJson(const std::string& jsonString, TypeOptions allowed) {
  int i = skipBlank(jsonString, 0);
  std::string value = jsonString.substr(i);

  JsonCompletion completion = completeAny(value, allowed, true);

  if (completion.index == 0 && completion.string.empty()) {
    throw MalformedJSONException("no valid JSON content found");
  }

  return value.substr(0, completion.index) + completion.string;
}

JsonCompletion completeAny(const std::string& jsonString,
                           TypeOptions allowed,
                           bool topLevel) {
  if (jsonString.empty()) {
    throw MalformedJSONException("empty string");
  }

  char firstChar = jsonString[0];

  switch (firstChar) {
    case '"':
      return completeString(jsonString, allowed);
    case '[':
      return completeArray(jsonString, allowed);
    case '{':
      return completeObject(jsonString, allowed);
    case '-':  // handles negative numbers
      if (jsonString.length() == 1) {
        throw MalformedJSONException("cannot parse singular '-'");
      } else if (jsonString.length() > 1 &&
                 jsonString[1] != 'I') {  // not negative infinity
        return completeNumber(jsonString, allowed, topLevel);
      }
      break;
    default:
      if (std::isdigit(static_cast<unsigned char>(firstChar))) {
        return completeNumber(jsonString, allowed, topLevel);
      }
      break;
  }

  // Handle NULL
  if (jsonString.substr(
          0, std::min(4, static_cast<int>(jsonString.length()))) == "null") {
    return JsonCompletion(4, "");
  }

  if (jsonString.length() < 4 &&
      std::string("null").substr(0, jsonString.length()) == jsonString) {
    if ((NULL_TYPE | allowed) == allowed) {
      return JsonCompletion(0, "null");
    }
    throw MalformedJSONException("cannot parse null with given options");
  }

  // Handle boolean true
  if (jsonString.substr(
          0, std::min(4, static_cast<int>(jsonString.length()))) == "true") {
    return JsonCompletion(4, "");
  }

  if (jsonString.length() < 4 &&
      std::string("true").substr(0, jsonString.length()) == jsonString) {
    if ((BOOL | allowed) == allowed) {
      return JsonCompletion(0, "true");
    }
    throw MalformedJSONException("cannot parse bool with given options");
  }

  // Handle boolean false
  if (jsonString.substr(
          0, std::min(5, static_cast<int>(jsonString.length()))) == "false") {
    return JsonCompletion(5, "");
  }

  if (jsonString.length() < 5 &&
      std::string("false").substr(0, jsonString.length()) == jsonString) {
    if ((BOOL | allowed) == allowed) {
      return JsonCompletion(0, "false");
    }
    throw MalformedJSONException("cannot parse bool with given options");
  }

  // Handle infinity
  if (jsonString.substr(0,
                        std::min(8, static_cast<int>(jsonString.length()))) ==
      "Infinity") {
    return JsonCompletion(8, "");
  }

  if (jsonString.length() < 8 &&
      std::string("Infinity").substr(0, jsonString.length()) == jsonString) {
    if ((INFINITY_TYPE | allowed) == allowed) {
      return JsonCompletion(0, "Infinity");
    }
    throw MalformedJSONException("cannot parse Infinity with given options");
  }

  // Handle negative infinity
  if (jsonString.substr(0,
                        std::min(9, static_cast<int>(jsonString.length()))) ==
      "-Infinity") {
    return JsonCompletion(9, "");
  }

  if (jsonString.length() < 9 &&
      std::string("-Infinity").substr(0, jsonString.length()) == jsonString) {
    if ((NEG_INFINITY | allowed) == allowed) {
      return JsonCompletion(0, "-Infinity");
    }
    throw MalformedJSONException("cannot parse -Infinity with given options");
  }

  // Handle NaN
  if (jsonString.substr(
          0, std::min(3, static_cast<int>(jsonString.length()))) == "NaN") {
    return JsonCompletion(3, "");
  }

  if (jsonString.length() < 3 &&
      std::string("NaN").substr(0, jsonString.length()) == jsonString) {
    if ((NAN_TYPE | allowed) == allowed) {
      return JsonCompletion(0, "NaN");
    }
    throw MalformedJSONException("cannot parse NaN with given options");
  }

  throw MalformedJSONException(std::string("MalformedJSON(unexpected char ") +
                               firstChar + ")");
}

// Matches Go completeString function exactly
JsonCompletion completeString(const std::string& jsonString,
                              TypeOptions allowed) {
  if (jsonString.empty() || jsonString[0] != '"') {
    throw MalformedJSONException("string must start with quote");
  }

  int index = 1;
  bool charEscaped = false;
  int stringLength = static_cast<int>(jsonString.length());

  while (index < stringLength && (jsonString[index] != '"' || charEscaped)) {
    if (jsonString[index] == '\\') {
      charEscaped = !charEscaped;
    } else {
      charEscaped = false;
    }
    index += 1;
  }

  if (index < stringLength) {
    return JsonCompletion(index + 1, "");
  }

  if ((STR | allowed) != allowed) {
    throw MalformedJSONException("cannot complete malformed json");
  }

  // Handle unicode and hex strings
  // Handle \uXXXX
  size_t u_index = jsonString.rfind("\\u");
  if (u_index != std::string::npos) {
    if (static_cast<int>(u_index) + 6 == stringLength) {
      return JsonCompletion(static_cast<int>(u_index) + 6, "\"");
    }
    return JsonCompletion(static_cast<int>(u_index) + 2, "\"");
  }

  // Handle \UXXXXXXXX
  size_t U_index = jsonString.rfind("\\U");
  if (U_index != std::string::npos) {
    if (static_cast<int>(U_index) + 10 == stringLength) {
      return JsonCompletion(static_cast<int>(U_index) + 10, "\"");
    }
    return JsonCompletion(static_cast<int>(U_index) + 2, "\"");
  }

  // Handle \xXX
  size_t x_index = jsonString.rfind("\\x");
  if (x_index != std::string::npos) {
    if (static_cast<int>(x_index) + 4 == stringLength) {
      return JsonCompletion(static_cast<int>(x_index) + 4, "\"");
    }
    return JsonCompletion(static_cast<int>(x_index) + 2, "\"");
  }

  if (charEscaped) {
    return JsonCompletion(index - 1, "\"");
  }

  return JsonCompletion(index, "\"");
}

JsonCompletion completeArray(const std::string& jsonString,
                             TypeOptions allowed) {
  int i = 1;
  int j = 1;
  int length = static_cast<int>(jsonString.length());

  while (j < length) {
    j = skipBlank(jsonString, j);
    if (j >= length) {
      break;
    }

    if (jsonString[j] == ']') {
      return JsonCompletion(j + 1, "");
    }

    try {
      JsonCompletion result = completeAny(jsonString.substr(j), allowed, false);

      // If the string in the result has some char in it, complete the array
      if (!result.string.empty()) {
        if ((ARR | allowed) == allowed) {
          return JsonCompletion(j + result.index, result.string + "]");
        }
        throw MalformedJSONException("cannot parse array with given options");
      }

      // First item in array is fine, check other items
      j += result.index;
      i = j;

      j = skipBlank(jsonString, j);
      if (j >= length) {
        break;
      }

      if (jsonString[j] == ',') {
        j += 1;
      } else if (jsonString[j] == ']') {
        return JsonCompletion(j + 1, "");
      } else {
        throw MalformedJSONException(
            std::string("MalformedJSON(expected \",\" or \"]\" got ") +
            jsonString[j] + ")");
      }
    } catch (const MalformedJSONException&) {
      // Can't complete the array, make it empty
      if ((ARR | allowed) == allowed) {
        return JsonCompletion(i, "]");
      }
      throw MalformedJSONException("cannot parse array with given options");
    }
  }

  // Reached end of string, close array at last known good point
  if ((ARR | allowed) == allowed) {
    return JsonCompletion(i, "]");
  }
  throw MalformedJSONException("cannot parse array with given options");
}

// Matches Go completeObject function exactly
JsonCompletion completeObject(const std::string& jsonString,
                              TypeOptions allowed) {
  int i = 1;
  int j = 1;
  int length = static_cast<int>(jsonString.length());

  while (j < length) {
    j = skipBlank(jsonString, j);
    if (j >= length) {
      break;
    }

    if (jsonString[j] == '}') {
      return JsonCompletion(j + 1, "");
    }

    try {
      JsonCompletion key = completeString(jsonString.substr(j), allowed);
      if (!key.string.empty()) {
        // Can't parse the key or key is incomplete
        if ((OBJ | allowed) == allowed) {
          return JsonCompletion(i, "}");
        }
        throw MalformedJSONException("cannot parse object with given options");
      }

      // Move index by key length
      j += key.index;

      j = skipBlank(jsonString, j);
      if (j >= length) {
        break;
      }

      if (jsonString[j] != ':') {
        throw MalformedJSONException(
            std::string("MalformedJSON( expected \":\" got ") + jsonString[j] +
            ")");
      }
      j += 1;

      j = skipBlank(jsonString, j);
      if (j >= length) {
        break;
      }

      JsonCompletion result = completeAny(jsonString.substr(j), allowed, false);

      // If the string in the result has some char in it, complete the object
      if (!result.string.empty()) {
        if ((OBJ | allowed) == allowed) {
          return JsonCompletion(j + result.index, result.string + "}");
        }
        throw MalformedJSONException("cannot parse object with given options");
      }

      // First key-value pair is fine, check other items
      j += result.index;
      i = j;

      j = skipBlank(jsonString, j);
      if (j >= length) {
        break;
      }

      if (jsonString[j] == ',') {
        j += 1;
      } else if (jsonString[j] == '}') {
        return JsonCompletion(j + 1, "");
      } else {
        throw MalformedJSONException(
            std::string("MalformedJSON(expected \",\" or \"}\" got ") +
            jsonString[j] + ")");
      }
    } catch (const MalformedJSONException& e) {
      // Check if this is a case where the key is not a valid string start
      // For cases like "{0", we should throw the exception
      if (j < length && jsonString[j] != '"' && jsonString[j] != '}') {
        // This is an invalid key (not starting with quote), re-throw exception
        throw e;
      }

      // For other cases (like incomplete but valid objects), return empty
      // object
      if ((OBJ | allowed) == allowed) {
        return JsonCompletion(i, "}");
      }
      throw MalformedJSONException("cannot parse object with given options");
    }
  }

  // Reached end of string, close object at last known good point
  if ((OBJ | allowed) == allowed) {
    return JsonCompletion(i, "}");
  }
  throw MalformedJSONException("cannot parse object with given options");
}

JsonCompletion completeNumber(const std::string& jsonString,
                              TypeOptions allowed,
                              bool topLevel) {
  int i = 1;
  int length = static_cast<int>(jsonString.length());

  // Move forwards while we still have numbers, including exponents and decimals
  while (i < length &&
         (std::isdigit(static_cast<unsigned char>(jsonString[i])) ||
          jsonString[i] == '.' || jsonString[i] == '+' ||
          jsonString[i] == '-' || jsonString[i] == 'e' ||
          jsonString[i] == 'E')) {
    i += 1;
  }

  bool specialNum = false;
  // no boundary check initially
  while (i >= 1 && (jsonString[i - 1] == '.' || jsonString[i - 1] == '-' ||
                    jsonString[i - 1] == '+' || jsonString[i - 1] == 'e' ||
                    jsonString[i - 1] == 'E')) {
    i -= 1;
    specialNum = true;
    // If we've gone to position 0, we need to check if we can continue
    if (i == 0) {
      throw MalformedJSONException("string index out of range");
    }
  }

  if (specialNum || (i == length && !topLevel)) {
    if ((NUM | allowed) == allowed) {
      return JsonCompletion(i, "");
    }
    throw MalformedJSONException("cannot parse number with given options");
  }

  return JsonCompletion(i, "");
}

}  // namespace partial_json_parser