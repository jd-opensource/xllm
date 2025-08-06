#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <memory>
#include "common/uuid.h"
namespace llm {
namespace function_call {

class BaseFormatDetector {
public:
    virtual ~BaseFormatDetector() = default;
    
    virtual bool detect(const std::string& text) const = 0;
    
    virtual FormatDetectionResult detect_format(const std::string& text) const = 0;
    
    virtual ParseResult parse_calls(const std::string& text) const = 0;
    
    virtual StreamingParseResult parse_streaming(const std::string& chunk) = 0;
    
    virtual void reset_streaming_state() = 0;
    
    virtual ModelFormat get_format() const = 0;
    
    virtual std::string get_format_name() const = 0;
    
    virtual StructureInfo get_structure_info() const = 0;
    
    virtual EBNFGrammar generate_ebnf_grammar(const ConstraintOptions& options = {}) const = 0;
    
    virtual bool validate_call_format(const ToolCallItem& call) const = 0;
    
protected:
    std::string generate_call_id() const {
        return "call_" + llm::generate_uuid();
    }
    
    std::string clean_json_string(const std::string& json_str) const {
        std::string cleaned = json_str;
        size_t start = cleaned.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        size_t end = cleaned.find_last_not_of(" \t\n\r");
        cleaned = cleaned.substr(start, end - start + 1);
        return cleaned;
    }
    
    bool is_valid_json(const std::string& json_str) const {
        try {
            int brace_count = 0;
            bool in_string = false;
            bool escaped = false;
            
            for (char c : json_str) {
                if (escaped) {
                    escaped = false;
                    continue;
                }
                
                if (c == '\\') {
                    escaped = true;
                    continue;
                }
                
                if (c == '"') {
                    in_string = !in_string;
                    continue;
                }
                
                if (!in_string) {
                    if (c == '{') brace_count++;
                    else if (c == '}') brace_count--;
                }
            }
            
            return brace_count == 0 && !in_string;
        } catch (...) {
            return false;
        }
    }
};

class DetectorFactory {
public:
    static std::unique_ptr<BaseFormatDetector> create_detector(ModelFormat format);
    
    static std::vector<std::unique_ptr<BaseFormatDetector>> create_all_detectors();
    
    static ModelFormat infer_format_from_model_name(const std::string& model_name);
    
    static std::string get_format_name(ModelFormat format);
    
private:
    DetectorFactory() = default;
};

}  // namespace function_call
}  // namespace llm