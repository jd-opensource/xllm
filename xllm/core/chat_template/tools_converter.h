#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>
#include "chat_template.h"

namespace llm {

class ToolsConverter {
 public:
  static std::string convert_tools_to_json(const std::vector<Tool>& tools);
  
  static std::string convert_tools_to_prompt(
      const std::vector<Tool>& tools,
      const std::string& tool_choice = "auto");
  
  static std::vector<ToolCall> parse_tool_calls_from_text(
      const std::string& model_output);
  
  static std::vector<ToolCall> parse_tool_calls_from_json(
      const std::string& json_str);
  
  static bool validate_tool_call_arguments(
      const ToolCall& tool_call,
      const std::vector<Tool>& available_tools);
  
  static std::string generate_tool_call_id();
  
  static std::string format_tool_choice(const std::string& tool_choice);
  
 private:
  static std::optional<ToolCall> parse_single_function_call(
      const nlohmann::json& json_obj);
  
  static bool validate_json_schema(
      const std::string& json_str,
      const std::string& schema_str);
  
  static std::string clean_json_string(const std::string& raw_json);
  
  static std::vector<std::string> extract_json_blocks(const std::string& text);
};

}  // namespace llm