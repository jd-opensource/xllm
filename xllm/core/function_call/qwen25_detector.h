#pragma once

#include "base_format_detector.h"
#include <string>
#include <string_view>
#include <regex>

namespace llm {
namespace function_call {

class Qwen25Detector : public BaseFormatDetector {
public:
    Qwen25Detector();
    
    virtual ~Qwen25Detector() = default;

private:
    std::string normal_text_buffer_;
    
    std::regex tool_call_regex_;
    
    std::string_view trim_whitespace(std::string_view str) const;
    
    std::vector<std::pair<size_t, size_t>> find_tool_call_ranges(const std::string& text) const;

public:
    bool has_tool_call(const std::string& text) override;

    StreamingParseResult detect_and_parse(const std::string& text, const std::vector<proto::Tool>& tools) override;
};

}  // namespace function_call
}  // namespace llm