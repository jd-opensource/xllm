#pragma once

#include <regex>
#include <string>

#include "base_format_detector.h"

namespace xllm {
namespace function_call {

/**
 * Detector for Kimi K2 model function call format.
 *
 * Format Structure:
 * ```
 * <|tool_calls_section_begin|>
 * <|tool_call_begin|>functions.{func_name}:{index}
 * <|tool_call_argument_begin|>{json_args}<|tool_call_end|>
 * <|tool_calls_section_end|>
 * ```
 *
 * Reference:
 * https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
 */
class KimiK2Detector : public BaseFormatDetector {
 public:
  KimiK2Detector();

  virtual ~KimiK2Detector() = default;

 private:
  std::string tool_call_start_token_;
  std::string tool_call_end_token_;
  std::string tool_call_argument_begin_token_;

  std::regex tool_call_regex_;

  std::string last_arguments_;

 public:
  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

 private:
  std::string extract_function_name(const std::string& tool_call_id) const;

  int extract_function_index(const std::string& tool_call_id) const;
};

}  // namespace function_call
}  // namespace xllm