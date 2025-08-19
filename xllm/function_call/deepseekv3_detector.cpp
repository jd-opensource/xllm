#include "deepseekv3_detector.h"

#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string_view>

namespace xllm {
namespace function_call {

DeepSeekV3Detector::DeepSeekV3Detector() : BaseFormatDetector() {
  bot_token_ = "<｜tool▁calls▁begin｜>";
  eot_token_ = "<｜tool▁calls▁end｜>";
  tool_call_separator_ = "";
}

bool DeepSeekV3Detector::has_tool_call(const std::string& text) {
  return text.find(bot_token_) != std::string::npos;
}

std::string_view DeepSeekV3Detector::trim_whitespace(
    std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string_view{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return str.substr(start, end - start + 1);
}

std::vector<std::pair<size_t, size_t>>
DeepSeekV3Detector::find_tool_call_ranges(const std::string& text) const {
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(4);

  const std::string call_begin = "<｜tool▁call▁begin｜>";
  const std::string call_end = "<｜tool▁call▁end｜>";

  size_t search_pos = 0;
  const size_t call_begin_len = call_begin.length();
  const size_t call_end_len = call_end.length();

  while (search_pos < text.length()) {
    size_t start_pos = text.find(call_begin, search_pos);
    if (start_pos == std::string::npos) {
      break;
    }

    size_t content_start = start_pos + call_begin_len;
    size_t end_pos = text.find(call_end, content_start);
    if (end_pos == std::string::npos) {
      break;
    }

    ranges.emplace_back(content_start, end_pos);
    search_pos = end_pos + call_end_len;
  }

  return ranges;
}

StreamingParseResult DeepSeekV3Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t bot_token_pos = text.find(bot_token_);

  std::string normal_text;
  if (bot_token_pos != std::string::npos) {
    std::string_view normal_text_view(text.data(), bot_token_pos);
    std::string_view trimmed = trim_whitespace(normal_text_view);
    normal_text = std::string(trimmed);
  } else {
    std::string_view trimmed = trim_whitespace(text);
    normal_text = std::string(trimmed);
    return StreamingParseResult(normal_text);
  }

  auto tool_call_ranges = find_tool_call_ranges(text);

  std::vector<ToolCallItem> calls;
  calls.reserve(tool_call_ranges.size());

  for (const auto& range : tool_call_ranges) {
    std::string_view content_view(text.data() + range.first,
                                  range.second - range.first);
    std::string_view trimmed_content = trim_whitespace(content_view);

    if (trimmed_content.empty()) {
      continue;
    }

    try {
      // Parse DeepSeek V3 format: <tool_sep>function_name\n```json\n{args}\n```
      const std::string tool_sep = "<｜tool▁sep｜>";
      size_t sep_pos = trimmed_content.find(tool_sep);
      if (sep_pos == std::string_view::npos) {
        LOG(ERROR) << "Failed to find tool separator in: "
                   << std::string(trimmed_content);
        continue;
      }

      // Extract function name (between tool_sep and first newline)
      size_t name_start = sep_pos + tool_sep.length();
      size_t name_end = trimmed_content.find('\n', name_start);
      if (name_end == std::string_view::npos) {
        LOG(ERROR) << "Failed to find function name end in: "
                   << std::string(trimmed_content);
        continue;
      }

      std::string_view func_name_view =
          trimmed_content.substr(name_start, name_end - name_start);
      std::string_view func_name_trimmed = trim_whitespace(func_name_view);
      std::string func_name(func_name_trimmed);

      // Find JSON block (between ```json\n and \n```)
      const std::string json_start = "```json\n";
      const std::string json_end = "\n```";

      size_t json_start_pos = trimmed_content.find(json_start, name_end);
      if (json_start_pos == std::string_view::npos) {
        LOG(ERROR) << "Failed to find JSON start in: "
                   << std::string(trimmed_content);
        continue;
      }

      size_t json_content_start = json_start_pos + json_start.length();
      size_t json_end_pos = trimmed_content.find(json_end, json_content_start);
      if (json_end_pos == std::string_view::npos) {
        LOG(ERROR) << "Failed to find JSON end in: "
                   << std::string(trimmed_content);
        continue;
      }

      std::string_view json_view = trimmed_content.substr(
          json_content_start, json_end_pos - json_content_start);
      std::string_view json_trimmed = trim_whitespace(json_view);

      // Parse JSON arguments
      nlohmann::json func_args;
      try {
        std::string json_content(json_trimmed);
        func_args = nlohmann::json::parse(json_content);
      } catch (const nlohmann::json::parse_error& e) {
        LOG(ERROR) << "Failed to parse JSON arguments: "
                   << std::string(json_trimmed)
                   << ", JSON parse error: " << e.what();
        continue;
      }

      // Create JSON object for parse_base_json
      nlohmann::json match_json;
      match_json["name"] = func_name;
      match_json["parameters"] = func_args;

      auto parsed_calls = parse_base_json(match_json, tools);
      calls.insert(calls.end(),
                   std::make_move_iterator(parsed_calls.begin()),
                   std::make_move_iterator(parsed_calls.end()));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to parse tool call: "
                 << std::string(trimmed_content) << ", error: " << e.what();
      continue;
    }
  }

  return StreamingParseResult(std::move(normal_text), std::move(calls));
}

}  // namespace function_call
}  // namespace xllm