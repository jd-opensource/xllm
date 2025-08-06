#include "function_call_parser.h"
#include "base_detector.h"
#include <algorithm>
#include <glog/logging.h>

namespace llm {
namespace function_call {

FunctionCallParser::FunctionCallParser() : preferred_format_(ModelFormat::UNKNOWN) {
    initialize_detectors();
}

void FunctionCallParser::initialize_detectors() {
    detectors_ = DetectorFactory::create_all_detectors();
    
    for (auto& detector : detectors_) {
        if (detector) {
            format_detectors_[detector->get_format()] = std::move(detector);
        }
    }
    detectors_.clear();
}

void FunctionCallParser::set_preferred_format(ModelFormat format) {
    preferred_format_ = format;
}

void FunctionCallParser::set_preferred_format(const std::string& model_name) {
    preferred_format_ = infer_format_from_model_name(model_name);
}

ParseResult FunctionCallParser::parse_auto(const std::string& text) {
    if (preferred_format_ != ModelFormat::UNKNOWN) {
        auto detector = get_detector(preferred_format_);
        if (detector && detector->detect(text)) {
            return detector->parse_calls(text);
        }
    }
    
    for (auto& [format, detector] : format_detectors_) {
        if (detector && detector->detect(text)) {
            return detector->parse_calls(text);
        }
    }
    
    return {};
}

ParseResult FunctionCallParser::parse_with_format(const std::string& text, ModelFormat format) {
    auto detector = get_detector(format);
    if (!detector) {
        LOG(WARNING) << "Unsupported format: " << static_cast<int>(format);
        return {};
    }
    
    return detector->parse_calls(text);
}

StreamingParseResult FunctionCallParser::parse_streaming_auto(const std::string& chunk) {
    if (preferred_format_ != ModelFormat::UNKNOWN) {
        auto detector = get_detector(preferred_format_);
        if (detector) {
            return detector->parse_streaming(chunk);
        }
    }
    
    for (auto& [format, detector] : format_detectors_) {
        if (detector) {
            auto result = detector->parse_streaming(chunk);
            if (result.has_completed_calls() || result.has_partial_call()) {
                return result;
            }
        }
    }
    
    StreamingParseResult empty_result;
    return empty_result;
}

StreamingParseResult FunctionCallParser::parse_streaming_with_format(const std::string& chunk, ModelFormat format) {
    auto detector = get_detector(format);
    if (!detector) {
        StreamingParseResult result;
        result.has_error = true;
        result.error_message = "Unsupported format";
        return result;
    }
    
    return detector->parse_streaming(chunk);
}

std::vector<FormatDetectionResult> FunctionCallParser::detect_formats(const std::string& text) {
    std::vector<FormatDetectionResult> results;
    
    for (auto& [format, detector] : format_detectors_) {
        if (detector) {
            auto result = detector->detect_format(text);
            results.push_back(result);
        }
    }
    
    std::sort(results.begin(), results.end(), 
              [](const FormatDetectionResult& a, const FormatDetectionResult& b) {
                  return a.confidence > b.confidence;
              });
    
    return results;
}

FormatDetectionResult FunctionCallParser::get_best_format(const std::string& text) {
    auto results = detect_formats(text);
    if (!results.empty()) {
        return results[0];
    }
    
    FormatDetectionResult empty_result;
    return empty_result;
}

bool FunctionCallParser::validate_calls(const std::vector<ToolCallItem>& calls, ModelFormat format) {
    auto detector = get_detector(format);
    if (!detector) {
        return false;
    }
    
    for (const auto& call : calls) {
        if (!detector->validate_call_format(call)) {
            return false;
        }
    }
    
    return true;
}

std::string FunctionCallParser::generate_constraints(const std::vector<std::string>& function_names, 
                                                    ModelFormat format,
                                                    const ConstraintOptions& options) {
    ModelFormat target_format = format;
    if (target_format == ModelFormat::UNKNOWN) {
        target_format = preferred_format_;
    }
    if (target_format == ModelFormat::UNKNOWN) {
        target_format = ModelFormat::QWEN25; // 默认格式
    }
    
    auto detector = get_detector(target_format);
    if (!detector) {
        return "";
    }
    
    ConstraintOptions modified_options = options;
    modified_options.allowed_functions = function_names;
    
    auto grammar = detector->generate_ebnf_grammar(modified_options);
    return grammar.to_string();
}

void FunctionCallParser::reset_all_streaming_states() {
    for (auto& [format, detector] : format_detectors_) {
        if (detector) {
            detector->reset_streaming_state();
        }
    }
}

void FunctionCallParser::reset_streaming_state(ModelFormat format) {
    auto detector = get_detector(format);
    if (detector) {
        detector->reset_streaming_state();
    }
}

std::vector<ModelFormat> FunctionCallParser::get_supported_formats() const {
    std::vector<ModelFormat> formats;
    for (const auto& [format, detector] : format_detectors_) {
        if (detector) {
            formats.push_back(format);
        }
    }
    return formats;
}

std::string FunctionCallParser::get_format_name(ModelFormat format) const {
    return DetectorFactory::get_format_name(format);
}

bool FunctionCallParser::is_format_supported(ModelFormat format) const {
    return format_detectors_.find(format) != format_detectors_.end();
}

BaseFormatDetector* FunctionCallParser::get_detector(ModelFormat format) {
    auto it = format_detectors_.find(format);
    if (it != format_detectors_.end()) {
        return it->second.get();
    }
    return nullptr;
}

ModelFormat FunctionCallParser::infer_format_from_model_name(const std::string& model_name) {
    return DetectorFactory::infer_format_from_model_name(model_name);
}

namespace utils {

std::vector<ToolCallItem> parse_function_calls(const std::string& text) {
    static FunctionCallParser parser;
    return parser.parse_auto(text).tool_calls;
}

std::vector<ToolCallItem> parse_function_calls(const std::string& text, const std::string& format) {
    static FunctionCallParser parser;
    
    ModelFormat model_format = ModelFormat::UNKNOWN;
    if (format == "qwen25" || format == "qwen") {
        model_format = ModelFormat::QWEN25;
    }

    if (model_format == ModelFormat::UNKNOWN) {
        return parser.parse_auto(text).tool_calls;
    }
    
    return parser.parse_with_format(text, model_format).tool_calls;
}

bool has_function_calls(const std::string& text) {
    static FunctionCallParser parser;
    auto calls = parser.parse_auto(text);
    return calls.has_tool_calls();
}

std::string detect_best_format(const std::string& text) {
    static FunctionCallParser parser;
    auto result = parser.get_best_format(text);
    return parser.get_format_name(result.format);
}

std::string generate_ebnf_constraints(const std::vector<std::string>& function_names, 
                                     const std::string& format) {
    static FunctionCallParser parser;
    
    ModelFormat model_format = ModelFormat::UNKNOWN;
    if (format == "qwen25" || format == "qwen") {
        model_format = ModelFormat::QWEN25;
    }
    
    return parser.generate_constraints(function_names, model_format);
}

bool validate_function_call_format(const ToolCallItem& call, const std::string& format) {
    static FunctionCallParser parser;
    
    ModelFormat model_format = ModelFormat::UNKNOWN;
    if (format == "qwen25" || format == "qwen") {
        model_format = ModelFormat::QWEN25;
    }
    
    if (model_format == ModelFormat::UNKNOWN) {
        return false;
    }
    
    return parser.validate_calls({call}, model_format);
}

}  // namespace utils

}  // namespace function_call
}  // namespace llm