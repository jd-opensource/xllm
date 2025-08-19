#pragma once

#include <string>

#include "base_format_detector.h"

namespace xllm {
namespace function_call {

class DeepSeekV3Detector : public BaseFormatDetector {
 public:
  DeepSeekV3Detector();

  virtual ~DeepSeekV3Detector() = default;

  bool has_tool_call(const std::string& text) override;

  StreamingParseResult detect_and_parse(
      const std::string& text,
      const std::vector<JsonTool>& tools) override;

 private:
  std::string func_call_regex_;
  std::string func_detail_regex_;
  std::string_view trim_whitespace(std::string_view str) const;
  std::vector<std::pair<size_t, size_t>> find_tool_call_ranges(
      const std::string& text) const;
};

}  // namespace function_call
}  // namespace xllm