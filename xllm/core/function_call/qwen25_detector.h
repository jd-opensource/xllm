#pragma once

#include "base_format_detector.h"
#include <string>

namespace llm {
namespace function_call {

class Qwen25Detector : public BaseFormatDetector {
public:
    Qwen25Detector();
    
    virtual ~Qwen25Detector() = default;

private:
    std::string normal_text_buffer_;  // Buffer for handling partial end tokens

public:

    bool has_tool_call(const std::string& text) override;

    StreamingParseResult detect_and_parse(const std::string& text, const std::vector<proto::Tool>& tools) override;
};

}  // namespace function_call
}  // namespace llm