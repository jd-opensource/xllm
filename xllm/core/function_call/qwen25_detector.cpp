#include "qwen25_detector.h"

#include <algorithm>
#include <iostream>
#include <string_view>

namespace llm {
namespace function_call {

Qwen25Detector::Qwen25Detector() : BaseFormatDetector() {
  bot_token_ = "<tool_call>\n";
  eot_token_ = "\n</tool_call>";
  tool_call_separator_ = "\n";

  std::string pattern = bot_token_ + "([\\s\\S]*?)" + eot_token_;
  tool_call_regex_ = std::regex(
      pattern,
      std::regex_constants::ECMAScript | std::regex_constants::optimize);
}

bool Qwen25Detector::has_tool_call(const std::string& text) {
  return text.find(bot_token_) != std::string::npos;
}

std::string_view Qwen25Detector::trim_whitespace(std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string_view{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return str.substr(start, end - start + 1);
}

std::vector<std::pair<size_t, size_t>> Qwen25Detector::find_tool_call_ranges(
    const std::string& text) const {
  std::vector<std::pair<size_t, size_t>> ranges;
  ranges.reserve(4);

  size_t search_pos = 0;
  const size_t bot_token_len = bot_token_.length();
  const size_t eot_token_len = eot_token_.length();

  while (search_pos < text.length()) {
    size_t start_pos = text.find(bot_token_, search_pos);
    if (start_pos == std::string::npos) {
      break;
    }

    size_t content_start = start_pos + bot_token_len;
    size_t end_pos = text.find(eot_token_, content_start);
    if (end_pos == std::string::npos) {
      break;
    }

    ranges.emplace_back(content_start, end_pos);
    search_pos = end_pos + eot_token_len;
  }

  return ranges;
}

StreamingParseResult Qwen25Detector::detect_and_parse(
    const std::string& text,
    const std::vector<proto::Tool>& tools) {
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
      std::string json_content(trimmed_content);
      auto json_obj = nlohmann::json::parse(json_content);
      auto parsed_calls = parse_base_json(json_obj, tools);

      calls.insert(calls.end(),
                   std::make_move_iterator(parsed_calls.begin()),
                   std::make_move_iterator(parsed_calls.end()));
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to parse JSON part: "
                 << std::string(trimmed_content)
                 << ", JSON parse error: " << e.what();
      continue;
    }
  }

  return StreamingParseResult(std::move(normal_text), std::move(calls));
}

}  // namespace function_call
}  // namespace llm