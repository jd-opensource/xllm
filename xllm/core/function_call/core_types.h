#pragma once

#include <string>
#include <vector>
#include <optional>
#include <functional>
#include "chat.pb.h"

namespace llm {
namespace function_call {

struct ToolCallItem {
    int tool_index;
    std::optional<std::string> name;
    std::string parameters;  // JSON string
    
    ToolCallItem() : tool_index(-1), parameters("") {}
    
    ToolCallItem(int index, const std::optional<std::string>& func_name, const std::string& params)
        : tool_index(index), name(func_name), parameters(params) {}
};

struct StreamingParseResult {
    std::string normal_text;
    std::vector<ToolCallItem> calls;
    
    StreamingParseResult() = default;
    
    StreamingParseResult(const std::string& text) : normal_text(text) {}
    
    StreamingParseResult(const std::vector<ToolCallItem>& tool_calls) : calls(tool_calls) {}
    
    StreamingParseResult(const std::string& text, const std::vector<ToolCallItem>& tool_calls) 
        : normal_text(text), calls(tool_calls) {}
    
    bool has_calls() const {
        return !calls.empty();
    }
    
    void clear() {
        normal_text.clear();
        calls.clear();
    }
};


struct StructureInfo {
    std::string begin;
    std::string end;
    std::string trigger;
    
    StructureInfo() = default;
    
    StructureInfo(const std::string& begin_str, const std::string& end_str, const std::string& trigger_str)
        : begin(begin_str), end(end_str), trigger(trigger_str) {}
};


using GetInfoFunc = std::function<StructureInfo(const std::string&)>;

}  // namespace function_call
}  // namespace llm