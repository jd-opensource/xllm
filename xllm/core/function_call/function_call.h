#pragma once

#include "base_format_detector.h"
#include "core_types.h"
#include "function_call_parser.h"
#include "qwen25_detector.h"
#include "kimik2_detector.h"
#include "deepseekv3_detector.h"

namespace llm {
namespace function_call {

inline std::vector<ToolCallItem> parse(const std::string& text,
                                       const std::vector<JsonTool>& tools,
                                       const std::string& format = "qwen25") {
  return utils::parse_function_calls(text, tools, format);
}

inline bool has_calls(const std::string& text,
                      const std::string& format = "qwen25") {
  return utils::has_function_calls(text, format);
}

}  // namespace function_call
}  // namespace llm