#include "function_call_parser.h"

#include <iostream>
#include <stdexcept>

#include "core/util/uuid.h"
#include "deepseekv3_detector.h"
#include "kimik2_detector.h"
#include "qwen25_detector.h"
namespace xllm {
namespace function_call {

const std::unordered_map<std::string, std::string>
    FunctionCallParser::ToolCallParserEnum = {
        {"qwen25", "qwen25"},
        {"qwen3", "qwen25"},
        {"kimi_k2", "kimi_k2"},
        {"deepseekv3", "deepseekv3"},
        // TODO
        // {"llama3", "llama3"},
        // {"mistral", "mistral"},
        // {"pythonic", "pythonic"},
        // {"qwen3_coder", "qwen3_coder"},
        // {"glm45", "glm45"},
        // {"step3", "step3"},
};

FunctionCallParser::FunctionCallParser(const std::vector<JsonTool>& tools,
                                       const std::string& tool_call_parser)
    : tools_(tools) {
  detector_ = create_detector(tool_call_parser);
  CHECK(detector_ != nullptr)
      << "Unsupported tool_call_parser: " << tool_call_parser
      << ". Supported parsers are: " << [this]() {
           std::string supported;
           for (const auto& [key, value] : ToolCallParserEnum) {
             if (!supported.empty()) supported += ", ";
             supported += key;
           }
           return supported;
         }();
}

bool FunctionCallParser::has_tool_call(const std::string& text) const {
  return detector_->has_tool_call(text);
}

std::tuple<std::string, std::vector<ToolCallItem>>
FunctionCallParser::parse_non_stream(const std::string& full_text) {
  StreamingParseResult parsed_result =
      detector_->detect_and_parse(full_text, tools_);

  if (!parsed_result.calls.empty()) {
    return std::make_tuple(parsed_result.normal_text, parsed_result.calls);
  } else {
    return std::make_tuple(full_text, std::vector<ToolCallItem>());
  }
}

std::unique_ptr<BaseFormatDetector> FunctionCallParser::create_detector(
    const std::string& tool_call_parser) {
  auto it = ToolCallParserEnum.find(tool_call_parser);
  if (it == ToolCallParserEnum.end()) {
    return nullptr;
  }

  if (it->second == "qwen25") {
    return std::make_unique<Qwen25Detector>();
  }

  if (it->second == "kimi_k2") {
    return std::make_unique<KimiK2Detector>();
  }

  if (it->second == "deepseekv3") {
    return std::make_unique<DeepSeekV3Detector>();
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
    const std::vector<JsonTool>& tools,
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

bool has_function_calls(const std::string& text,
                        const std::string& parser_type) {
  try {
    FunctionCallParser parser({}, parser_type);
    return parser.has_tool_call(text);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error checking function calls: " << e.what();
    return false;
  }
}

thread_local ShortUUID short_uuid;

std::string generate_tool_call_id() { return "call_" + short_uuid.random(); }

}  // namespace utils

}  // namespace function_call
}  // namespace xllm