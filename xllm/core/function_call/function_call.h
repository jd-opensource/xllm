#pragma once

#include "types.h"
#include "function_call_parser.h"

namespace llm {
namespace function_call {

class FunctionCallInterface {
public:
    static FunctionCallInterface& getInstance() {
        static FunctionCallInterface instance;
        return instance;
    }
    
    ParseResult parse(const std::string& text) {
        return parser_.parse_auto(text);
    }
    
    ParseResult parse(const std::string& text, const std::string& format) {
        ModelFormat model_format = ModelFormat::UNKNOWN;
        if (format == "qwen" || format == "qwen25") {
            model_format = ModelFormat::QWEN25;
        }
        return parser_.parse_with_format(text, model_format);
    }
    
    StreamingParseResult parseStreaming(const std::string& chunk) {
        return parser_.parse_streaming_auto(chunk);
    }
    
    bool hasFunction(const std::string& text) {
        return utils::has_function_calls(text);
    }
    
    std::string detectFormat(const std::string& text) {
        return utils::detect_best_format(text);
    }
    
    std::string generateConstraints(const std::vector<std::string>& function_names, 
                                   const std::string& format = "auto") {
        return utils::generate_ebnf_constraints(function_names, format);
    }
    
    void setPreferredFormat(const std::string& model_name) {
        parser_.set_preferred_format(model_name);
    }
    
    void resetStreamingState() {
        parser_.reset_all_streaming_states();
    }
    
private:
    FunctionCallInterface() = default;
    FunctionCallParser parser_;
};

}  // namespace function_call

inline function_call::ParseResult parse_function_calls(const std::string& text) {
    return function_call::FunctionCallInterface::getInstance().parse(text);
}

inline bool has_function_calls(const std::string& text) {
    auto result = function_call::FunctionCallInterface::getInstance().parse(text);
    return result.has_tool_calls();
}

}  // namespace llm