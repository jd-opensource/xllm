/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "function_call_parser.h"

#include <iostream>
#include <stdexcept>

#include "core/util/uuid.h"
#include "deepseekv3_detector.h"
#include "glm45_detector.h"
#include "glm47_detector.h"
#include "kimik2_detector.h"
#include "qwen25_detector.h"
namespace xllm {
namespace function_call {

const std::unordered_map<std::string, std::string>
    FunctionCallParser::kToolCallParserMap = {
        {"qwen25", "qwen25"},
        {"qwen3", "qwen25"},
        {"kimi_k2", "kimi_k2"},
        {"deepseekv3", "deepseekv3"},
        {"glm45", "glm45"},
        {"glm47", "glm47"},
        // TODO
        // {"llama3", "llama3"},
        // {"mistral", "mistral"},
        // {"pythonic", "pythonic"},
        // {"qwen3_coder", "qwen3_coder"},
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
           for (const auto& [key, value] : kToolCallParserMap) {
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

StreamingParseResult FunctionCallParser::parse_streaming_increment(
    const std::string& new_text) {
  return detector_->parse_streaming_increment(new_text, tools_);
}

std::unique_ptr<BaseFormatDetector> FunctionCallParser::create_detector(
    const std::string& tool_call_parser) {
  auto it = kToolCallParserMap.find(tool_call_parser);
  if (it == kToolCallParserMap.end()) {
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

  if (it->second == "glm45") {
    return std::make_unique<Glm45Detector>();
  }

  if (it->second == "glm47") {
    return std::make_unique<Glm47Detector>();
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

StreamingParseResult parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools,
    const std::string& parser_type) {
  try {
    FunctionCallParser parser(tools, parser_type);
    return parser.parse_streaming_increment(new_text);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in streaming parsing: " << e.what();
    return StreamingParseResult();
  }
}

thread_local ShortUUID short_uuid;

std::string generate_tool_call_id() { return "call_" + short_uuid.random(); }

}  // namespace utils

}  // namespace function_call
}  // namespace xllm