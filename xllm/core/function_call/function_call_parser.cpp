#include "function_call_parser.h"
#include <stdexcept>
#include <iostream>

namespace llm {
namespace function_call {

const std::unordered_map<std::string, std::string> FunctionCallParser::ToolCallParserEnum = {
    {"qwen25", "qwen25"},
    {"qwen3", "qwen25"},
    // TODO
    // {"llama3", "llama3"},
    // {"mistral", "mistral"},
    // {"deepseekv3", "deepseekv3"},
    // {"pythonic", "pythonic"},
    // {"kimi_k2", "kimi_k2"},
    // {"qwen3_coder", "qwen3_coder"},
    // {"glm45", "glm45"},
    // {"step3", "step3"},
};

FunctionCallParser::FunctionCallParser(const std::vector<proto::Tool>& tools, const std::string& tool_call_parser)
    : tools_(tools) {
    
    detector_ = create_detector(tool_call_parser);
    if (!detector_) {
        throw std::invalid_argument("Unsupported tool_call_parser: " + tool_call_parser);
    }
}

bool FunctionCallParser::has_tool_call(const std::string& text) const {
    return detector_->has_tool_call(text);
}

std::tuple<std::string, std::vector<ToolCallItem>> FunctionCallParser::parse_non_stream(const std::string& full_text) {
    StreamingParseResult parsed_result = detector_->detect_and_parse(full_text, tools_);
    
    if (!parsed_result.calls.empty()) {
        return std::make_tuple(parsed_result.normal_text, parsed_result.calls);
    } else {
        return std::make_tuple(full_text, std::vector<ToolCallItem>());
    }
}


std::unique_ptr<BaseFormatDetector> FunctionCallParser::create_detector(const std::string& tool_call_parser) {
    auto it = ToolCallParserEnum.find(tool_call_parser);
    if (it == ToolCallParserEnum.end()) {
        return nullptr;
    }
    
    if (it->second == "qwen25") {
        return std::make_unique<Qwen25Detector>();
    }
    
    // if (tool_call_parser == "llama3") {
    //     return std::make_unique<Llama32Detector>();
    // }
    // if (tool_call_parser == "mistral") {
    //     return std::make_unique<MistralDetector>();
    // }
    
    return nullptr;
}

namespace utils {

std::vector<ToolCallItem> parse_function_calls(
    const std::string& text, 
    const std::vector<proto::Tool>& tools,
    const std::string& parser_type) {
    
    try {
        FunctionCallParser parser(tools, parser_type);
        auto [normal_text, calls] = parser.parse_non_stream(text);
        return calls;
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error parsing function calls: " << e.what();
        return {};
    }
}


bool has_function_calls(
    const std::string& text,
    const std::string& parser_type) {
    
    try {
        FunctionCallParser parser({}, parser_type);
        return parser.has_tool_call(text);
    } catch (const std::exception& e) {
        LOG(ERROR) << "Error checking function calls: " << e.what();
        return false;
    }
}

}  // namespace utils

}  // namespace function_call
}  // namespace llm