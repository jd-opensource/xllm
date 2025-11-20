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

#include "jinja_chat_template.h"

#include <glog/logging.h>
#include <unistd.h>

#include <optional>
#include <string>

namespace xllm {

JinjaChatTemplate::JinjaChatTemplate(const TokenizerArgs& args) : args_(args) {
  try {
    template_ = std::make_unique<minja::chat_template>(
        args_.chat_template(), args_.bos_token(), args_.eos_token());
    LOG(INFO) << "Jinja chat template init succeed.";

  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse jinja chat template, TokenizerArgs: "
               << args_ << std::endl
               << "Error message: " << e.what();
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json chat_template_kwargs;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  const std::vector<xllm::JsonTool> empty_tools;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages) const {
  // Call the overloaded method with empty tools
  nlohmann::ordered_json empty_tools = nlohmann::json::array();
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;

    if (std::holds_alternative<std::string>(message.content)) {
      message_json["content"] = std::get<std::string>(message.content);
    } else if (std::holds_alternative<Message::MMContentVec>(message.content)) {
      message_json["content"] =
          get_mm_content(std::get<Message::MMContentVec>(message.content));
    }

    if (message.tool_call_id.has_value()) {
      message_json["tool_call_id"] = message.tool_call_id.value();
    }

    if (message.reasoning_content.has_value()) {
      message_json["reasoning_content"] = message.reasoning_content.value();
    }

    if (message.tool_calls.has_value()) {
      nlohmann::ordered_json tool_calls_json = nlohmann::json::array();
      for (const auto& tool_call : message.tool_calls.value()) {
        nlohmann::ordered_json tool_call_json;
        tool_call_json["id"] = tool_call.id;
        tool_call_json["type"] = tool_call.type;

        nlohmann::ordered_json function_json;
        function_json["name"] = tool_call.function.name;
        function_json["arguments"] = tool_call.function.arguments;
        tool_call_json["function"] = function_json;

        tool_calls_json.push_back(tool_call_json);
      }
      message_json["tool_calls"] = tool_calls_json;
    }

    messages_json.push_back(message_json);
  }
  std::cerr << "Complete messages_json: " << messages_json.dump(2, ' ', false)
            << std::endl;

  nlohmann::ordered_json tools_json = nlohmann::json::array();
  for (const auto& json_tool : json_tools) {
    nlohmann::ordered_json tool_json;
    tool_json["type"] = json_tool.type;

    nlohmann::ordered_json function_json;
    function_json["name"] = json_tool.function.name;
    function_json["description"] = json_tool.function.description;
    function_json["parameters"] = json_tool.function.parameters;

    tool_json["function"] = function_json;
    tools_json.push_back(tool_json);
  }
  // apply the template
  return apply(messages_json, tools_json, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  minja::chat_template_inputs input;
  input.messages = messages;
  input.tools = tools;
  input.add_generation_prompt = true;
  input.extra_context = chat_template_kwargs;
  minja::chat_template_options options;

  auto result = template_->apply(input, options);
  std::cerr << "JinjaChatTemplate apply result: " << result << std::endl;

  return result;
}

nlohmann::ordered_json JinjaChatTemplate::get_mm_content(
    const Message::MMContentVec& vec) const {
  nlohmann::ordered_json content_json = nlohmann::json::array();

  for (const auto& item : vec) {
    nlohmann::ordered_json item_json;
    item_json["type"] = item.type;

    if (item.type == "text") {
      item_json["text"] = item.text;
    } else {
      item_json[item.type] = "mm place holder";
    }

    content_json.emplace_back(item_json);
  }

  return std::move(content_json);
}

}  // namespace xllm
