#include "qwen25_detector.h"
#include <regex>
#include <iostream>

namespace llm {
namespace function_call {

Qwen25Detector::Qwen25Detector() : BaseFormatDetector() {
    bot_token_ = "<tool_call>\n";
    eot_token_ = "\n</tool_call>";
    tool_call_separator_ = "\n";
}

bool Qwen25Detector::has_tool_call(const std::string& text) {
    return text.find(bot_token_) != std::string::npos;
}

StreamingParseResult Qwen25Detector::detect_and_parse(const std::string& text, const std::vector<proto::Tool>& tools) {
    size_t idx = text.find(bot_token_);
    std::string normal_text = (idx != std::string::npos) ? text.substr(0, idx) : text;
    
    while (!normal_text.empty() && std::isspace(normal_text.back())) {
        normal_text.pop_back();
    }
    
    if (text.find(bot_token_) == std::string::npos) {
        return StreamingParseResult(normal_text);
    }

    std::string escaped_bot_token = bot_token_;
    std::string escaped_eot_token = eot_token_;
    
    std::string pattern = escaped_bot_token + "(.*?)" + escaped_eot_token;
    
    size_t pos = 0;
    while ((pos = pattern.find("\n", pos)) != std::string::npos) {
        pattern.replace(pos, 1, "\\n");
        pos += 2;
    }
    
    std::regex tool_call_regex(pattern, std::regex_constants::ECMAScript);
    std::sregex_iterator iter(text.begin(), text.end(), tool_call_regex);
    std::sregex_iterator end;
    
    std::vector<ToolCallItem> calls;
    
    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        std::string match_result = match[1].str();
        
        try {
            match_result.erase(0, match_result.find_first_not_of(" \t\n\r"));
            match_result.erase(match_result.find_last_not_of(" \t\n\r") + 1);
            
            auto parsed_calls = parse_base_json(match_result, tools);
            calls.insert(calls.end(), parsed_calls.begin(), parsed_calls.end());
        } catch (const std::exception& e) {
            LOG(ERROR) << "Failed to parse JSON part: " << match_result
                      << ", JSON parse error: " << e.what();
            continue;
        }
    }
    
    return StreamingParseResult(normal_text, calls);
}

}  // namespace function_call
}  // namespace llm