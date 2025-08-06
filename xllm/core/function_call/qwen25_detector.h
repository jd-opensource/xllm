#pragma once

#include "base_detector.h"
#include <regex>
#include <nlohmann/json.hpp>

namespace llm {
namespace function_call {

class Qwen25Detector : public BaseFormatDetector {
private:
    std::regex start_pattern_;
    std::regex end_pattern_;
    std::regex full_pattern_;
    std::string buffer_;
    
public:
    Qwen25Detector() 
        : start_pattern_(R"(<tool_call>)"),
          end_pattern_(R"(</tool_call>)"),
          full_pattern_(R"(<tool_call>\s*(\{.*?\})\s*</tool_call>)", 
                       std::regex_constants::ECMAScript) {}
    
    ~Qwen25Detector() override = default;
    
    bool detect(const std::string& text) const override {
        return std::regex_search(text, start_pattern_) && 
               std::regex_search(text, end_pattern_);
    }
    
    FormatDetectionResult detect_format(const std::string& text) const override {
        FormatDetectionResult result;
        result.format = ModelFormat::QWEN25;
        
        bool has_start = std::regex_search(text, start_pattern_);
        bool has_end = std::regex_search(text, end_pattern_);
        
        if (has_start && has_end) {
            result.confidence = 0.95;
            result.reason = "Found complete <tool_call> tags";
        } else if (has_start) {
            result.confidence = 0.7;
            result.reason = "Found opening <tool_call> tag";
        } else {
            result.confidence = 0.0;
            result.reason = "No Qwen2.5 format markers found";
        }
        
        result.structure_info = get_structure_info();
        return result;
    }
    
    ParseResult parse_calls(const std::string& text) const override {
        ParseResult result;
        std::string normal_text = text;
        
        std::sregex_iterator iter(text.begin(), text.end(), full_pattern_);
        std::sregex_iterator end;
        
        std::vector<std::pair<size_t, size_t>> match_positions;
        
        for (; iter != end; ++iter) {
            const std::smatch& match = *iter;
            std::string json_content = match[1].str();
            
            match_positions.push_back({match.position(), match.length()});
            
            ToolCallItem call;
            call.id = generate_call_id();
            call.type = "function";
            
            if (parse_json_content(json_content, call)) {
                call.state = ParseState::COMPLETED;
                result.tool_calls.push_back(call);
            } else {
                call.state = ParseState::ERROR;
                call.error = "Failed to parse JSON content: " + json_content;
                result.tool_calls.push_back(call);
            }
        }
        
        for (auto it = match_positions.rbegin(); it != match_positions.rend(); ++it) {
            normal_text.erase(it->first, it->second);
        }
        
        result.normal_text = normal_text;
        return result;
    }
    
    StreamingParseResult parse_streaming(const std::string& chunk) override {
        buffer_ += chunk;
        StreamingParseResult result;
        
        auto completed_result = parse_calls(buffer_);
        result.completed_calls = completed_result.tool_calls;
        result.normal_text = completed_result.normal_text;
        
        if (std::regex_search(buffer_, start_pattern_) && 
            !std::regex_search(buffer_, end_pattern_)) {
            ToolCallItem partial;
            partial.id = "partial_call";
            partial.type = "function";
            partial.state = ParseState::PARSING;
            result.partial_call = partial;
        }
        
        if (!completed_result.tool_calls.empty()) {
            size_t last_end = buffer_.rfind("</tool_call>");
            if (last_end != std::string::npos) {
                last_end += 12; // Length of "</tool_call>"
                result.remaining_text = buffer_.substr(last_end);
                buffer_ = result.remaining_text;
            }
        }
        
        return result;
    }
    
    void reset_streaming_state() override {
        buffer_.clear();
    }
    
    ModelFormat get_format() const override {
        return ModelFormat::QWEN25;
    }
    
    std::string get_format_name() const override {
        return "qwen25";
    }
    
    StructureInfo get_structure_info() const override {
        StructureInfo info("qwen25", "<tool_call>", "</tool_call>");
        info.patterns["function_call"] = R"(<tool_call>\s*\{.*?\}\s*</tool_call>)";
        info.patterns["json_content"] = R"(\{.*?\})";
        return info;
    }
    
    EBNFGrammar generate_ebnf_grammar(const ConstraintOptions& options) const override {
        EBNFGrammar grammar;
        grammar.start_rule = "tool_calls";
        
        if (options.allow_multiple_calls) {
            grammar.add_rule(EBNFRule("tool_calls", "tool_call+"));
        } else {
            grammar.add_rule(EBNFRule("tool_calls", "tool_call"));
        }
        
        grammar.add_rule(EBNFRule("tool_call", "\"<tool_call>\" ws json_object ws \"</tool_call>\""));
        
        grammar.add_rule(EBNFRule("json_object", "\"{\" ws json_members ws \"}\""));
        grammar.add_rule(EBNFRule("json_members", "json_member (\",\" ws json_member)*"));
        grammar.add_rule(EBNFRule("json_member", "json_string \":\" ws json_value"));
        
        grammar.add_rule(EBNFRule("json_value", "json_string | json_object | json_array"));
        grammar.add_rule(EBNFRule("json_string", "\"\\\"\" [^\"\\\\]* \"\\\"\""));
        grammar.add_rule(EBNFRule("json_array", "\"[\" ws (json_value (\",\" ws json_value)*)? ws \"]\""));
        
        grammar.add_rule(EBNFRule("ws", "[ \\t\\n\\r]*", true));
        
        return grammar;
    }
    
    bool validate_call_format(const ToolCallItem& call) const override {
        if (call.function_name.empty()) return false;
        if (call.function_arguments.empty()) return false;
        return is_valid_json(call.function_arguments);
    }
    
private:
    bool parse_json_content(const std::string& json_str, ToolCallItem& call) const {
        try {
            auto json_obj = nlohmann::json::parse(clean_json_string(json_str));
            
            if (json_obj.contains("name") && json_obj["name"].is_string()) {
                call.function_name = json_obj["name"].get<std::string>();
            } else {
                return false;
            }
            
            if (json_obj.contains("arguments")) {
                if (json_obj["arguments"].is_string()) {
                    call.function_arguments = json_obj["arguments"].get<std::string>();
                } else {
                    call.function_arguments = json_obj["arguments"].dump();
                }
            } else {
                call.function_arguments = "{}";
            }
            
            return true;
        } catch (const nlohmann::json::exception& e) {
            call.error = "JSON parse error: " + std::string(e.what());
            return false;
        }
    }
};

}  // namespace function_call
}  // namespace llm