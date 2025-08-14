#include "kimik2_detector.h"

#include <iostream>
#include <regex>
#include <stdexcept>

namespace llm {
namespace function_call {

KimiK2Detector::KimiK2Detector() : BaseFormatDetector() {
  // Initialize KimiK2 specific tokens
  bot_token_ = "<|tool_calls_section_begin|>";
  eot_token_ = "<|tool_calls_section_end|>";
  tool_call_start_token_ = "<|tool_call_begin|>";
  tool_call_end_token_ = "<|tool_call_end|>";
  tool_call_argument_begin_token_ = "<|tool_call_argument_begin|>";

  // Regex pattern for parsing tool calls with the following format:
  //   <|tool_call_begin|>functions.{func_name}:{index}
  //   <|tool_call_argument_begin|>{json_args}<|tool_call_end|>
  // Note: C++ regex doesn't support named groups, so we use numbered groups:
  //   Group 1: tool_call_id (functions.{func_name}:{index})
  //   Group 2: function_arguments ({json_args})
  std::string pattern =
      R"(<\|tool_call_begin\|>\s*([\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>)";

  try {
    tool_call_regex_ = std::regex(pattern, std::regex_constants::ECMAScript);
  } catch (const std::regex_error& e) {
    LOG(ERROR) << "Failed to compile KimiK2 regex pattern: " << e.what();
    throw;
  }

  last_arguments_ = "";
}

bool KimiK2Detector::has_tool_call(const std::string& text) {
  // Check if the text contains the KimiK2 tool call section begin token
  return text.find(bot_token_) != std::string::npos;
}

StreamingParseResult KimiK2Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t bot_pos = text.find(bot_token_);

  std::string normal_text =
      (bot_pos != std::string::npos) ? text.substr(0, bot_pos) : text;

  if (bot_pos == std::string::npos) {
    return StreamingParseResult(normal_text);
  }

  std::vector<ToolCallItem> calls;

  try {
    std::sregex_iterator iter(
        text.begin() + bot_pos, text.end(), tool_call_regex_);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
      std::smatch match = *iter;

      if (match.size() >= 3) {
        std::string tool_call_id = match[1].str();
        std::string function_arguments = match[2].str();

        std::string function_name = extract_function_name(tool_call_id);
        int function_index = extract_function_index(tool_call_id);

        calls.emplace_back(function_index,     // Use the call index in the
                                               // response, not tool position
                           function_name,      // Function name
                           function_arguments  // JSON parameters
        );
      }
    }

    return StreamingParseResult(normal_text, calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in KimiK2 detect_and_parse: " << e.what();
    // Return the normal text if parsing fails
    return StreamingParseResult(normal_text);
  }
}

std::string KimiK2Detector::extract_function_name(
    const std::string& tool_call_id) const {
  // tool_call_id format: functions.{func_name}:{index}
  // Example: functions.get_weather:0

  try {
    // Find the position of "functions."
    size_t functions_pos = tool_call_id.find("functions.");
    if (functions_pos == std::string::npos) {
      LOG(WARNING)
          << "Invalid tool_call_id format, missing 'functions.' prefix: "
          << tool_call_id;
      return "";
    }

    // Skip "functions." (10 characters)
    size_t start_pos = functions_pos + 10;

    // Find the position of the last colon
    size_t colon_pos = tool_call_id.find_last_of(':');
    if (colon_pos == std::string::npos || colon_pos <= start_pos) {
      LOG(WARNING) << "Invalid tool_call_id format, missing ':' separator: "
                   << tool_call_id;
      return "";
    }

    // Extract function name between "functions." and ":"
    return tool_call_id.substr(start_pos, colon_pos - start_pos);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error extracting function name from tool_call_id: "
               << tool_call_id << ", error: " << e.what();
    return "";
  }
}

int KimiK2Detector::extract_function_index(
    const std::string& tool_call_id) const {
  // tool_call_id format: functions.{func_name}:{index}
  // Example: functions.get_weather:0

  try {
    // Find the position of the last colon
    size_t colon_pos = tool_call_id.find_last_of(':');
    if (colon_pos == std::string::npos) {
      LOG(WARNING) << "Invalid tool_call_id format, missing ':' separator: "
                   << tool_call_id;
      return 0;
    }

    // Extract index string after the colon
    std::string index_str = tool_call_id.substr(colon_pos + 1);

    // Convert to integer
    return std::stoi(index_str);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error extracting function index from tool_call_id: "
               << tool_call_id << ", error: " << e.what();
    return 0;
  }
}

}  // namespace function_call
}  // namespace llm