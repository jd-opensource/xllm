#include "jinja_chat_template.h"

#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
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
  const std::vector<proto::Tool> empty_tools;
  return apply(messages, empty_tools);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages, const std::vector<Tool>& tools) const {
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

    messages_json.push_back(message_json);
  }
  
  // convert tools to json object
  nlohmann::ordered_json tools_json = nlohmann::json::array();
  if (!tools.empty()) {
    try {
      // Use ToolsConverter to convert tools to JSON string, then parse it
      std::string tools_json_str = ToolsConverter::convert_tools_to_json(tools);
      tools_json = nlohmann::json::parse(tools_json_str);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to convert tools to JSON: " << e.what();
      // Continue with empty tools array
    }
  }
  
  // apply the template with tools
  return apply(messages_json, tools_json);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages) const {
  // Call the overloaded method with empty tools
  nlohmann::ordered_json empty_tools = nlohmann::json::array();
  return apply(messages, empty_tools);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools) const {
  minja::chat_template_inputs input;
  input.messages = messages;
  input.tools = tools;
  input.add_generation_prompt = true;
  minja::chat_template_options options;

  return template_->apply(input, options);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<proto::Tool>& proto_tools) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;
    message_json["content"] = message.content;
    messages_json.push_back(message_json);
  }

  // convert protobuf tools to json object
  nlohmann::ordered_json tools_json = nlohmann::json::array();
  if (!proto_tools.empty()) {
    try {
      for (const auto& proto_tool : proto_tools) {
        nlohmann::ordered_json tool_json;
        tool_json["type"] = proto_tool.type();

        nlohmann::ordered_json function_json;
        function_json["name"] = proto_tool.function().name();
        function_json["description"] = proto_tool.function().description();

        if (proto_tool.function().has_parameters()) {
          std::string parameters_json_str;
          google::protobuf::util::JsonPrintOptions options;
          options.add_whitespace = false;
          options.preserve_proto_field_names = true;
          auto status = google::protobuf::util::MessageToJsonString(
              proto_tool.function().parameters(),
              &parameters_json_str,
              options);
          if (status.ok()) {
            function_json["parameters"] =
                nlohmann::json::parse(parameters_json_str);
          } else {
            LOG(WARNING) << "Failed to convert parameters Struct to JSON: "
                         << status.message()
                         << ", tool: " << proto_tool.function().name();
            function_json["parameters"] = nlohmann::json::object();
          }
        } else {
          function_json["parameters"] = nlohmann::json::object();
        }

        tool_json["function"] = function_json;
        tools_json.push_back(tool_json);
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to convert protobuf tools to JSON: " << e.what();
      // Continue with empty tools array
      tools_json = nlohmann::json::array();
    }
  }

  // apply the template with tools
  return apply(messages_json, tools_json);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<proto::Tool>& proto_tools) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;
    message_json["content"] = message.content;
    messages_json.push_back(message_json);
  }

  // convert protobuf tools to json object
  nlohmann::ordered_json tools_json = nlohmann::json::array();
  if (!proto_tools.empty()) {
    try {
      for (const auto& proto_tool : proto_tools) {
        nlohmann::ordered_json tool_json;
        tool_json["type"] = proto_tool.type();

        nlohmann::ordered_json function_json;
        function_json["name"] = proto_tool.function().name();
        function_json["description"] = proto_tool.function().description();

        if (proto_tool.function().has_parameters()) {
          std::string parameters_json_str;
          google::protobuf::util::JsonPrintOptions options;
          options.add_whitespace = false;
          options.preserve_proto_field_names = true;
          auto status = google::protobuf::util::MessageToJsonString(
              proto_tool.function().parameters(),
              &parameters_json_str,
              options);
          if (status.ok()) {
            function_json["parameters"] =
                nlohmann::json::parse(parameters_json_str);
          } else {
            LOG(WARNING) << "Failed to convert parameters Struct to JSON: "
                         << status.message()
                         << ", tool: " << proto_tool.function().name();
            function_json["parameters"] = nlohmann::json::object();
          }
        } else {
          function_json["parameters"] = nlohmann::json::object();
        }

        tool_json["function"] = function_json;
        tools_json.push_back(tool_json);
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to convert protobuf tools to JSON: " << e.what();
      // Continue with empty tools array
      tools_json = nlohmann::json::array();
    }
  }

  // apply the template with tools
  return apply(messages_json, tools_json);
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
