#pragma once

#include "core_types.h"
#include "chat.pb.h"
#include <glog/logging.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace llm {
namespace function_call {

class BaseFormatDetector {
public:
    BaseFormatDetector();
    virtual ~BaseFormatDetector() = default;

    BaseFormatDetector(const BaseFormatDetector&) = delete;
    BaseFormatDetector& operator=(const BaseFormatDetector&) = delete;

protected:
    // Streaming state management
    // Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
    std::string buffer_;
    
    // Stores complete tool call info (name and arguments) for each tool being parsed.
    // Used by serving layer for completion handling when streaming ends.
    // Format: [{"name": str, "arguments": dict}, ...]
    std::vector<std::unordered_map<std::string, std::string>> prev_tool_call_arr_;
    
    // Index of currently streaming tool call. Starts at -1 (no active tool),
    // increments as each tool completes. Tracks which tool's arguments are streaming.
    int current_tool_id_;
    
    // Flag for whether current tool's name has been sent to client.
    // Tool names sent first with empty parameters, then arguments stream incrementally.
    bool current_tool_name_sent_;
    
    // Tracks raw JSON string content streamed to client for each tool's arguments.
    // Critical for serving layer to calculate remaining content when streaming ends.
    // Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
    std::vector<std::string> streamed_args_for_tool_;

    // Token configuration (override in subclasses)
    std::string bot_token_;
    std::string eot_token_;
    std::string tool_call_separator_;

    // Tool indices cache
    std::unordered_map<std::string, int> tool_indices_;

public:
    std::unordered_map<std::string, int> get_tool_indices(const std::vector<proto::Tool>& tools);

    std::vector<ToolCallItem> parse_base_json(const std::string& action_json, const std::vector<proto::Tool>& tools);

    virtual StreamingParseResult detect_and_parse(const std::string& text, const std::vector<proto::Tool>& tools) = 0;

    virtual bool has_tool_call(const std::string& text) = 0;
};

}  // namespace function_call
}  // namespace llm