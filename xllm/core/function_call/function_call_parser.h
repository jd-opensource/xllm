#pragma once

#include "types.h"
#include "base_detector.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace llm {
namespace function_call {

class FunctionCallParser {
private:
    std::vector<std::unique_ptr<BaseFormatDetector>> detectors_;
    std::unordered_map<ModelFormat, std::unique_ptr<BaseFormatDetector>> format_detectors_;
    ModelFormat preferred_format_;
    
public:
    FunctionCallParser();
    ~FunctionCallParser() = default;
    
    FunctionCallParser(const FunctionCallParser&) = delete;
    FunctionCallParser& operator=(const FunctionCallParser&) = delete;
    
    void set_preferred_format(ModelFormat format);
    void set_preferred_format(const std::string& model_name);
    
    ParseResult parse_auto(const std::string& text);
    
    ParseResult parse_with_format(const std::string& text, ModelFormat format);
    
    StreamingParseResult parse_streaming_auto(const std::string& chunk);
    
    StreamingParseResult parse_streaming_with_format(const std::string& chunk, ModelFormat format);
    
    std::vector<FormatDetectionResult> detect_formats(const std::string& text);
    
    FormatDetectionResult get_best_format(const std::string& text);
    
    bool validate_calls(const std::vector<ToolCallItem>& calls, ModelFormat format);
    
    std::string generate_constraints(const std::vector<std::string>& function_names, 
                                   ModelFormat format = ModelFormat::UNKNOWN,
                                   const ConstraintOptions& options = {});
    
    void reset_all_streaming_states();
    
    void reset_streaming_state(ModelFormat format);
    
    std::vector<ModelFormat> get_supported_formats() const;
    
    std::string get_format_name(ModelFormat format) const;
    
    bool is_format_supported(ModelFormat format) const;
    
private:
    void initialize_detectors();
    
    BaseFormatDetector* get_detector(ModelFormat format);
    
    ModelFormat infer_format_from_model_name(const std::string& model_name);
};

namespace utils {

std::vector<ToolCallItem> parse_function_calls(const std::string& text);

std::vector<ToolCallItem> parse_function_calls(const std::string& text, const std::string& format);

bool has_function_calls(const std::string& text);

std::string detect_best_format(const std::string& text);

std::string generate_ebnf_constraints(const std::vector<std::string>& function_names, 
                                     const std::string& format = "auto");

bool validate_function_call_format(const ToolCallItem& call, const std::string& format);

}  // namespace utils

}  // namespace function_call
}  // namespace llm