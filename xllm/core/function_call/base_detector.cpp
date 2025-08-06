#include "base_detector.h"
#include "qwen25_detector.h"
#include <algorithm>
#include <cctype>

namespace llm {
namespace function_call {

std::unique_ptr<BaseFormatDetector> DetectorFactory::create_detector(ModelFormat format) {
    switch (format) {
        case ModelFormat::QWEN25:
            return std::make_unique<Qwen25Detector>();
        default:
            return nullptr;
    }
}

std::vector<std::unique_ptr<BaseFormatDetector>> DetectorFactory::create_all_detectors() {
    std::vector<std::unique_ptr<BaseFormatDetector>> detectors;

    detectors.push_back(std::make_unique<Qwen25Detector>());

    return detectors;
}

ModelFormat DetectorFactory::infer_format_from_model_name(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    if (lower_name.find("qwen") != std::string::npos) {
        return ModelFormat::QWEN25;
    }
    
    return ModelFormat::UNKNOWN;
}

std::string DetectorFactory::get_format_name(ModelFormat format) {
    switch (format) {
        case ModelFormat::QWEN25:
            return "qwen25";
        default:
            return "unknown";
    }
}

}  // namespace function_call
}  // namespace llm