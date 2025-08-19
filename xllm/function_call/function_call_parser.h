#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "base_format_detector.h"
#include "core_types.h"

namespace xllm {
namespace function_call {

class FunctionCallParser {
 public:
  static const std::unordered_map<std::string, std::string> ToolCallParserEnum;

 private:
  std::unique_ptr<BaseFormatDetector> detector_;
  std::vector<JsonTool> tools_;

 public:
  FunctionCallParser(const std::vector<JsonTool>& tools,
                     const std::string& tool_call_parser);

  ~FunctionCallParser() = default;

  FunctionCallParser(const FunctionCallParser&) = delete;
  FunctionCallParser& operator=(const FunctionCallParser&) = delete;

  bool has_tool_call(const std::string& text) const;

  std::tuple<std::string, std::vector<ToolCallItem>> parse_non_stream(
      const std::string& full_text);

  // StructuralTagResponseFormat get_structure_tag();

  // std::tuple<std::string, std::any> get_structure_constraint(const
  // std::string& tool_choice);

  BaseFormatDetector* get_detector() const { return detector_.get(); }

 private:
  std::unique_ptr<BaseFormatDetector> create_detector(
      const std::string& tool_call_parser);
};

namespace utils {

std::vector<ToolCallItem> parse_function_calls(
    const std::string& text,
    const std::vector<JsonTool>& tools,
    const std::string& parser_type = "qwen25");

bool has_function_calls(const std::string& text,
                        const std::string& parser_type = "qwen25");

std::string generate_tool_call_id();
}  // namespace utils

}  // namespace function_call
}  // namespace xllm