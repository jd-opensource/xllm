#pragma once

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <memory>

namespace llm {
namespace function_call {

enum class ParseState {
    PENDING,
    PARSING,
    COMPLETED,
    ERROR
};

struct ToolCallItem {
    std::string id;
    std::string type;
    std::string function_name;
    std::string function_arguments;
    ParseState state = ParseState::PENDING;
    std::optional<std::string> error;
    
    ToolCallItem() = default;
    ToolCallItem(const std::string& id, const std::string& type, 
                 const std::string& name, const std::string& args)
        : id(id), type(type), function_name(name), function_arguments(args) {}
    
    bool is_valid() const {
        return state == ParseState::COMPLETED && 
               !function_name.empty() && 
               !function_arguments.empty() &&
               !error.has_value();
    }
    
    std::string to_string() const {
        std::string result = "ToolCallItem{";
        result += "id='" + id + "', ";
        result += "type='" + type + "', ";
        result += "function_name='" + function_name + "', ";
        result += "arguments='" + function_arguments + "', ";
        result += "state=" + std::to_string(static_cast<int>(state));
        if (error.has_value()) {
            result += ", error='" + error.value() + "'";
        }
        result += "}";
        return result;
    }
};

struct StreamingParseResult {
    std::string normal_text;
    std::vector<ToolCallItem> completed_calls;
    std::optional<ToolCallItem> partial_call;
    std::string remaining_text;
    bool has_error = false;
    std::string error_message;
    
    bool has_completed_calls() const {
        return !completed_calls.empty();
    }
    
    bool has_partial_call() const {
        return partial_call.has_value();
    }
    
    void clear() {
        completed_calls.clear();
        partial_call.reset();
        remaining_text.clear();
        has_error = false;
        error_message.clear();
    }
};
struct ParseResult {
    std::string normal_text;
    std::vector<ToolCallItem> tool_calls;
    bool has_error = false;
    std::string error_message;

    bool has_tool_calls() const {
        return !tool_calls.empty();
    }
    
    void clear() {
        normal_text.clear();
        tool_calls.clear();
        has_error = false;
        error_message.clear();
    }
};

struct StructureInfo {
    std::string format_name;
    std::string start_marker;
    std::string end_marker;
    std::unordered_map<std::string, std::string> patterns;

    StructureInfo() = default;
    StructureInfo(const std::string& name, const std::string& start, const std::string& end)
        : format_name(name), start_marker(start), end_marker(end) {}
};

enum class ModelFormat {
    QWEN25,
    UNKNOWN
};

struct FormatDetectionResult {
    ModelFormat format = ModelFormat::UNKNOWN;
    double confidence = 0.0;
    std::string reason;
    StructureInfo structure_info;
    
    bool is_valid() const {
        return format != ModelFormat::UNKNOWN && confidence > 0.5;
    }
};

struct EBNFRule {
    std::string name;
    std::string definition;
    bool is_terminal = false;
    
    EBNFRule() = default;
    EBNFRule(const std::string& name, const std::string& def, bool terminal = false)
        : name(name), definition(def), is_terminal(terminal) {}
};

struct EBNFGrammar {
    std::vector<EBNFRule> rules;
    std::string start_rule;
    
    void add_rule(const EBNFRule& rule) {
        rules.push_back(rule);
    }
    
    std::string to_string() const {
        std::string result;
        for (const auto& rule : rules) {
            result += rule.name + " ::= " + rule.definition + "\n";
        }
        return result;
    }
};

struct ConstraintOptions {
    bool allow_multiple_calls = true;
    bool require_arguments = true;
    bool strict_json = true;
    std::vector<std::string> allowed_functions;
    
    ConstraintOptions() = default;
};

}  // namespace function_call
}  // namespace llm