#include "base_format_detector.h"
#include <sstream>
#include <regex>
#include <iostream>

namespace llm {
namespace function_call {

BaseFormatDetector::BaseFormatDetector() 
    : current_tool_id_(-1)
    , current_tool_name_sent_(false)
    , bot_token_("")
    , eot_token_("")
    , tool_call_separator_(", ") {
}

std::unordered_map<std::string, int> BaseFormatDetector::get_tool_indices(const std::vector<proto::Tool>& tools) {
    std::unordered_map<std::string, int> indices;
    for (size_t i = 0; i < tools.size(); ++i) {
        if (!tools[i].function().name().empty()) {
            indices[tools[i].function().name()] = static_cast<int>(i);
        }
    }
    return indices;
}

std::vector<ToolCallItem> BaseFormatDetector::parse_base_json(const std::string& action_json, const std::vector<proto::Tool>& tools) {
    auto tool_indices = get_tool_indices(tools);
    std::vector<ToolCallItem> results;

    // TODO: Replace with a more robust JSON library for better functionality and reliability
    std::string trimmed = action_json;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);
    
    if (trimmed.empty()) {
        return results;
    }
    
    std::regex name_regex("\"name\"\\s*:\\s*\"([^\"]+)\"");
    std::regex args_regex("\"(?:parameters|arguments)\"\\s*:\\s*(\\{[^}]*\\})");
    
    std::smatch name_match, args_match;
    
    if (std::regex_search(trimmed, name_match, name_regex)) {
        std::string name = name_match[1].str();
        
        if (tool_indices.find(name) != tool_indices.end()) {
            std::string parameters = "{}";
            
            if (std::regex_search(trimmed, args_match, args_regex)) {
                parameters = args_match[1].str();
            }
            
            results.emplace_back(-1, name, parameters);
        } else {
            LOG(ERROR) << "Model attempted to call undefined function: " << name;
        }
    }
    
    return results;
}


}  // namespace function_call
}  // namespace llm