/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/chat_template/deepseek_v4_cpp_template.h"

#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace xllm {
namespace {

constexpr const char* kBosToken = "<｜begin▁of▁sentence｜>";
constexpr const char* kEosToken = "<｜end▁of▁sentence｜>";
constexpr const char* kThinkingStartToken = "<think>";
constexpr const char* kThinkingEndToken = "</think>";
constexpr const char* kDsmlToken = "｜DSML｜";
constexpr const char* kUserToken = "<｜User｜>";
constexpr const char* kAssistantToken = "<｜Assistant｜>";
constexpr const char* kLatestReminderToken = "<｜latest_reminder｜>";

constexpr const char* kToolsTemplate =
    "## Tools\n\n"
    "You have access to a set of tools to help answer the user's question. "
    "You can invoke tools by writing a \"<{dsml_token}tool_calls>\" block "
    "like the following:\n\n"
    "<{dsml_token}tool_calls>\n"
    "<{dsml_token}invoke name=\"$TOOL_NAME\">\n"
    "<{dsml_token}parameter "
    "name=\"$PARAMETER_NAME\" "
    "string=\"true|false\">"
    "$PARAMETER_VALUE"
    "</{dsml_token}parameter>\n"
    "...\n"
    "</{dsml_token}invoke>\n"
    "<{dsml_token}invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</{dsml_token}invoke>\n"
    "</{dsml_token}tool_calls>\n\n"
    "String parameters should be specified as is and set `string=\"true\"`. "
    "For all other types (numbers, booleans, arrays, objects), pass the value "
    "in JSON format and set `string=\"false\"`.\n\n"
    "If thinking_mode is enabled (triggered by {thinking_start_token}), you "
    "MUST output your complete reasoning inside "
    "{thinking_start_token}...{thinking_end_token} BEFORE any tool calls or "
    "final response.\n\n"
    "Otherwise, output directly after {thinking_end_token} with tool calls or "
    "final response.\n\n"
    "### Available Tool Schemas\n\n"
    "{tool_schemas}\n\n"
    "You MUST strictly follow the above defined tool name and parameter "
    "schemas to invoke tool calls.\n";

constexpr const char* kResponseFormatTemplate =
    "## Response Format:\n\n"
    "You MUST strictly adhere to the following schema to reply:\n"
    "{schema}";
constexpr const char* kToolCallTemplate =
    "<{dsml_token}invoke name=\"{name}\">\n{arguments}\n</{dsml_token}invoke>";
constexpr const char* kToolCallsTemplate =
    "<{dsml_token}tool_calls>\n{tool_calls}\n</{dsml_token}tool_calls>";
constexpr const char* kToolOutputTemplate =
    "<tool_result>{content}</tool_result>";
constexpr const char* kReasoningEffortMax =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose "
    "the problem to resolve the root cause, rigorously stress-testing your "
    "logic against all potential paths, edge cases, and adversarial "
    "scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every "
    "intermediate step, considered alternative, and rejected hypothesis to "
    "ensure absolutely no assumption is left unchecked.\n\n";

constexpr const char* kRoleSystem = "system";
constexpr const char* kRoleDeveloper = "developer";
constexpr const char* kRoleUser = "user";
constexpr const char* kRoleTool = "tool";
constexpr const char* kRoleAssistant = "assistant";
constexpr const char* kRoleLatestReminder = "latest_reminder";
constexpr const char* kRoleDirectSearchResults = "direct_search_results";

constexpr const char* kThinkingModeThinking = "thinking";
constexpr const char* kThinkingModeChat = "chat";

std::string to_json(const nlohmann::ordered_json& value) {
  try {
    return value.dump(/*indent=*/-1,
                      /*indent_char=*/' ',
                      /*ensure_ascii=*/false,
                      nlohmann::json::error_handler_t::replace);
  } catch (const std::exception&) {
    return value.dump(/*indent=*/-1,
                      /*indent_char=*/' ',
                      /*ensure_ascii=*/true,
                      nlohmann::json::error_handler_t::replace);
  }
}

nlohmann::ordered_json parse_json_object_or_empty(const std::string& text) {
  if (text.empty()) {
    return nlohmann::ordered_json::object();
  }
  try {
    nlohmann::ordered_json parsed = nlohmann::json::parse(text);
    if (parsed.is_object()) {
      return parsed;
    }
    return nlohmann::ordered_json::object();
  } catch (const std::exception&) {
    nlohmann::ordered_json fallback = nlohmann::ordered_json::object();
    fallback["arguments"] = text;
    return fallback;
  }
}

bool get_thinking_enabled(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("thinking") && kwargs["thinking"].is_boolean()) {
    return kwargs["thinking"].get<bool>();
  }
  if (kwargs.contains("enable_thinking") &&
      kwargs["enable_thinking"].is_boolean()) {
    return kwargs["enable_thinking"].get<bool>();
  }
  return false;
}

std::string get_thinking_mode(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("thinking_mode") && kwargs["thinking_mode"].is_string()) {
    return kwargs["thinking_mode"].get<std::string>();
  }
  return get_thinking_enabled(kwargs) ? kThinkingModeThinking
                                      : kThinkingModeChat;
}

std::string get_reasoning_effort(const nlohmann::ordered_json& kwargs) {
  if (kwargs.contains("reasoning_effort") &&
      kwargs["reasoning_effort"].is_string()) {
    return kwargs["reasoning_effort"].get<std::string>();
  }
  return "";
}

nlohmann::ordered_json openai_tools_to_functions(
    const nlohmann::ordered_json& tools) {
  nlohmann::ordered_json functions = nlohmann::ordered_json::array();
  if (!tools.is_array()) {
    return functions;
  }
  for (const auto& tool : tools) {
    if (tool.contains("function")) {
      functions.emplace_back(tool["function"]);
    }
  }
  return functions;
}

std::string render_tools(const nlohmann::ordered_json& tools) {
  std::vector<std::string> schemas;
  schemas.reserve(tools.size());
  for (const auto& tool : tools) {
    schemas.emplace_back(to_json(tool));
  }
  return absl::StrReplaceAll(std::string(kToolsTemplate),
                             {{"{tool_schemas}", absl::StrJoin(schemas, "\n")},
                              {"{dsml_token}", kDsmlToken},
                              {"{thinking_start_token}", kThinkingStartToken},
                              {"{thinking_end_token}", kThinkingEndToken}});
}

std::string render_response_format(
    const nlohmann::ordered_json& response_format) {
  return absl::StrReplaceAll(std::string(kResponseFormatTemplate),
                             {{"{schema}", to_json(response_format)}});
}

std::string get_text_content(const Message::Content& content) {
  if (std::holds_alternative<std::string>(content)) {
    return std::get<std::string>(content);
  }
  return "";
}

nlohmann::ordered_json tool_calls_from_openai_format(
    const Message::ToolCallVec& tool_calls) {
  nlohmann::ordered_json out = nlohmann::ordered_json::array();
  for (const Message::ToolCall& tool_call : tool_calls) {
    nlohmann::ordered_json item;
    item["id"] = tool_call.id;
    item["name"] = tool_call.function.name;
    item["arguments"] =
        parse_json_object_or_empty(tool_call.function.arguments);
    out.emplace_back(std::move(item));
  }
  return out;
}

nlohmann::ordered_json normalize_messages(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& tools) {
  nlohmann::ordered_json out = nlohmann::ordered_json::array();
  for (const Message& message : messages) {
    nlohmann::ordered_json item;
    item["role"] = message.role;
    item["content"] = get_text_content(message.content);
    if (message.tool_call_id.has_value()) {
      item["tool_call_id"] = message.tool_call_id.value();
    }
    if (message.reasoning_content.has_value()) {
      item["reasoning_content"] = message.reasoning_content.value();
    }
    if (message.tool_calls.has_value()) {
      item["tool_calls"] =
          tool_calls_from_openai_format(message.tool_calls.value());
    }
    out.emplace_back(std::move(item));
  }
  if (!tools.empty()) {
    nlohmann::ordered_json tools_json = nlohmann::ordered_json::array();
    for (const xllm::JsonTool& tool : tools) {
      nlohmann::ordered_json openai_tool;
      openai_tool["type"] = "function";
      nlohmann::ordered_json function;
      function["name"] = tool.function.name;
      function["description"] = tool.function.description;
      function["parameters"] = tool.function.parameters;
      openai_tool["function"] = std::move(function);
      tools_json.emplace_back(std::move(openai_tool));
    }

    nlohmann::ordered_json system_message;
    system_message["role"] = kRoleSystem;
    system_message["tools"] = std::move(tools_json);
    out.insert(out.begin(), std::move(system_message));
  }
  return out;
}

int32_t find_last_user_index(const nlohmann::ordered_json& messages) {
  for (int32_t idx = static_cast<int32_t>(messages.size()) - 1; idx >= 0;
       --idx) {
    std::string role = messages[idx].value("role", "");
    if (role == kRoleUser || role == kRoleDeveloper) {
      return idx;
    }
  }
  return -1;
}

nlohmann::ordered_json drop_thinking_messages(
    const nlohmann::ordered_json& messages,
    int32_t last_user_index) {
  nlohmann::ordered_json output = nlohmann::ordered_json::array();
  for (int32_t idx = 0; idx < static_cast<int32_t>(messages.size()); ++idx) {
    const nlohmann::ordered_json& message = messages[idx];
    std::string role = message.value("role", "");
    if (role == kRoleUser || role == kRoleSystem || role == kRoleTool ||
        role == kRoleLatestReminder || role == kRoleDirectSearchResults ||
        idx >= last_user_index) {
      output.emplace_back(message);
      continue;
    }
    if (role == kRoleAssistant) {
      nlohmann::ordered_json copied = message;
      copied.erase("reasoning_content");
      output.emplace_back(std::move(copied));
    }
  }
  return output;
}

std::string encode_arguments_to_dsml(const nlohmann::ordered_json& tool_call) {
  std::ostringstream oss;
  nlohmann::ordered_json arguments = nlohmann::ordered_json::object();
  if (tool_call.contains("arguments")) {
    arguments = tool_call["arguments"];
    if (arguments.is_string()) {
      arguments = parse_json_object_or_empty(arguments.get<std::string>());
    }
  }
  bool first = true;
  for (auto it = arguments.begin(); it != arguments.end(); ++it) {
    if (!first) {
      oss << "\n";
    }
    first = false;
    const nlohmann::ordered_json& value = it.value();
    bool is_string = value.is_string();
    oss << "<" << kDsmlToken << "parameter name=\"" << it.key()
        << "\" string=\"" << (is_string ? "true" : "false") << "\">";
    if (is_string) {
      oss << value.get<std::string>();
    } else {
      oss << to_json(value);
    }
    oss << "</" << kDsmlToken << "parameter>";
  }
  return oss.str();
}

std::string render_assistant_tool_calls(
    const nlohmann::ordered_json& tool_calls) {
  if (!tool_calls.is_array() || tool_calls.empty()) {
    return "";
  }
  std::vector<std::string> chunks;
  chunks.reserve(tool_calls.size());
  for (const auto& tool_call : tool_calls) {
    std::string invoke = absl::StrReplaceAll(
        std::string(kToolCallTemplate),
        {{"{dsml_token}", kDsmlToken},
         {"{name}", tool_call.value("name", "")},
         {"{arguments}", encode_arguments_to_dsml(tool_call)}});
    chunks.emplace_back(std::move(invoke));
  }
  std::string rendered =
      absl::StrReplaceAll(std::string(kToolCallsTemplate),
                          {{"{dsml_token}", kDsmlToken},
                           {"{tool_calls}", absl::StrJoin(chunks, "\n")}});
  return "\n\n" + rendered;
}

std::optional<int32_t> find_tool_call_order(
    const nlohmann::ordered_json& messages,
    int32_t tool_index) {
  std::string tool_call_id = messages[tool_index].value("tool_call_id", "");
  for (int32_t idx = tool_index - 1; idx >= 0; --idx) {
    const nlohmann::ordered_json& message = messages[idx];
    if (message.value("role", "") != kRoleAssistant) {
      continue;
    }
    if (!message.contains("tool_calls") || !message["tool_calls"].is_array()) {
      continue;
    }
    const nlohmann::ordered_json& tool_calls = message["tool_calls"];
    if (!tool_call_id.empty()) {
      for (int32_t tc_idx = 0; tc_idx < static_cast<int32_t>(tool_calls.size());
           ++tc_idx) {
        if (tool_calls[tc_idx].value("id", "") == tool_call_id) {
          return tc_idx;
        }
      }
    }
    int32_t sequential_order = tool_index - idx - 1;
    if (sequential_order >= 0 &&
        sequential_order < static_cast<int32_t>(tool_calls.size())) {
      return sequential_order;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

nlohmann::ordered_json merge_tool_messages(
    const nlohmann::ordered_json& messages) {
  nlohmann::ordered_json merged = nlohmann::ordered_json::array();
  for (int32_t idx = 0; idx < static_cast<int32_t>(messages.size()); ++idx) {
    const nlohmann::ordered_json& message = messages[idx];
    std::string role = message.value("role", "");
    if (role == kRoleTool) {
      nlohmann::ordered_json tool_block;
      tool_block["type"] = "tool_result";
      tool_block["tool_use_id"] = message.value("tool_call_id", "");
      tool_block["content"] = message.value("content", "");
      if (auto order = find_tool_call_order(messages, idx); order.has_value()) {
        tool_block["order"] = order.value();
      }
      if (!merged.empty() && merged.back().value("role", "") == kRoleUser &&
          merged.back().contains("content_blocks") &&
          merged.back()["content_blocks"].is_array()) {
        merged.back()["content_blocks"].emplace_back(std::move(tool_block));
      } else {
        nlohmann::ordered_json user_message;
        user_message["role"] = kRoleUser;
        user_message["content"] = "";
        user_message["content_blocks"] = nlohmann::ordered_json::array();
        user_message["content_blocks"].emplace_back(std::move(tool_block));
        merged.emplace_back(std::move(user_message));
      }
      continue;
    }

    if (role == kRoleUser) {
      nlohmann::ordered_json text_block;
      text_block["type"] = "text";
      text_block["text"] = message.value("content", "");
      if (!merged.empty() && merged.back().value("role", "") == kRoleUser &&
          merged.back().contains("content_blocks") &&
          merged.back()["content_blocks"].is_array()) {
        merged.back()["content_blocks"].emplace_back(std::move(text_block));
        continue;
      }
      nlohmann::ordered_json copied = message;
      copied["content_blocks"] = nlohmann::ordered_json::array();
      copied["content_blocks"].emplace_back(std::move(text_block));
      merged.emplace_back(std::move(copied));
      continue;
    }
    nlohmann::ordered_json copied = message;
    merged.emplace_back(std::move(copied));
  }
  return merged;
}

void sort_tool_results_by_call_order(nlohmann::ordered_json* messages) {
  if (messages == nullptr || !messages->is_array()) {
    return;
  }
  for (auto& message : *messages) {
    if (message.value("role", "") != kRoleUser ||
        !message.contains("content_blocks") ||
        !message["content_blocks"].is_array()) {
      continue;
    }
    auto& blocks = message["content_blocks"];
    std::vector<nlohmann::ordered_json> tool_blocks;
    for (const auto& block : blocks) {
      if (block.value("type", "") == "tool_result") {
        tool_blocks.emplace_back(block);
      }
    }
    std::stable_sort(tool_blocks.begin(),
                     tool_blocks.end(),
                     [](const auto& lhs, const auto& rhs) {
                       return lhs.value("order", 0) < rhs.value("order", 0);
                     });
    int32_t sorted_index = 0;
    for (auto& block : blocks) {
      if (block.value("type", "") == "tool_result") {
        block = tool_blocks[sorted_index++];
      }
    }
  }
}

std::string render_user_content(const nlohmann::ordered_json& msg) {
  if (!msg.contains("content_blocks") || !msg["content_blocks"].is_array()) {
    return msg.value("content", "");
  }
  std::vector<std::string> parts;
  for (const auto& block : msg["content_blocks"]) {
    std::string type = block.value("type", "");
    if (type == "text") {
      parts.emplace_back(block.value("text", ""));
    } else if (type == "tool_result") {
      parts.emplace_back(
          absl::StrReplaceAll(std::string(kToolOutputTemplate),
                              {{"{content}", block.value("content", "")}}));
    } else {
      parts.emplace_back("[Unsupported " + type + "]");
    }
  }
  return absl::StrJoin(parts, "\n\n");
}

std::string render_message(const nlohmann::ordered_json& messages,
                           int32_t index,
                           const std::string& thinking_mode,
                           bool drop_thinking,
                           const std::string& reasoning_effort) {
  if (index < 0 || index >= static_cast<int32_t>(messages.size())) {
    throw std::runtime_error("Message index out of range");
  }
  if (thinking_mode != kThinkingModeThinking &&
      thinking_mode != kThinkingModeChat) {
    throw std::runtime_error("Invalid thinking mode");
  }

  std::string prompt;
  const nlohmann::ordered_json& msg = messages[index];
  int32_t last_user_idx = find_last_user_index(messages);
  std::string role = msg.value("role", "");
  std::string content = msg.value("content", "");

  if (index == 0 && thinking_mode == kThinkingModeThinking &&
      reasoning_effort == "max") {
    prompt += kReasoningEffortMax;
  }

  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  if (msg.contains("tools")) {
    tools = msg["tools"];
  }
  nlohmann::ordered_json response_format;
  if (msg.contains("response_format")) {
    response_format = msg["response_format"];
  }

  if (role == kRoleSystem) {
    prompt += content;
    nlohmann::ordered_json fn_tools = openai_tools_to_functions(tools);
    if (!fn_tools.empty()) {
      prompt += "\n\n" + render_tools(fn_tools);
    }
    if (!response_format.is_null()) {
      prompt += "\n\n" + render_response_format(response_format);
    }
  } else if (role == kRoleDeveloper) {
    if (content.empty()) {
      throw std::runtime_error("Developer message content is empty");
    }
    prompt += kUserToken;
    prompt += content;
    nlohmann::ordered_json fn_tools = openai_tools_to_functions(tools);
    if (!fn_tools.empty()) {
      prompt += "\n\n" + render_tools(fn_tools);
    }
    if (!response_format.is_null()) {
      prompt += "\n\n" + render_response_format(response_format);
    }
  } else if (role == kRoleUser) {
    prompt += kUserToken;
    prompt += render_user_content(msg);
  } else if (role == kRoleLatestReminder) {
    prompt += kLatestReminderToken;
    prompt += content;
  } else if (role == kRoleAssistant) {
    std::string thinking_part;
    nlohmann::ordered_json tool_calls = nlohmann::ordered_json::array();
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
      tool_calls = msg["tool_calls"];
    }
    if (thinking_mode == kThinkingModeThinking) {
      if (!drop_thinking || index > last_user_idx) {
        thinking_part = msg.value("reasoning_content", "");
        thinking_part += kThinkingEndToken;
      }
    }
    prompt += thinking_part;
    prompt += content;
    prompt += render_assistant_tool_calls(tool_calls);
    prompt += kEosToken;
  } else {
    throw std::runtime_error("Unknown role: " + role);
  }

  if (index + 1 < static_cast<int32_t>(messages.size())) {
    std::string next_role = messages[index + 1].value("role", "");
    if (next_role != kRoleAssistant && next_role != kRoleLatestReminder) {
      return prompt;
    }
  }

  if (role == kRoleUser || role == kRoleDeveloper) {
    prompt += kAssistantToken;
    if ((!drop_thinking && thinking_mode == kThinkingModeThinking) ||
        (drop_thinking && thinking_mode == kThinkingModeThinking &&
         index >= last_user_idx)) {
      prompt += kThinkingStartToken;
    } else {
      prompt += kThinkingEndToken;
    }
  }
  return prompt;
}

bool has_tools(const nlohmann::ordered_json& messages) {
  for (const auto& message : messages) {
    if (message.contains("tools") && message["tools"].is_array() &&
        !message["tools"].empty()) {
      return true;
    }
  }
  return false;
}

}  // namespace

DeepseekV4CppTemplate::DeepseekV4CppTemplate(const TokenizerArgs& args)
    : args_(args) {}

std::optional<std::string> DeepseekV4CppTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json kwargs = nlohmann::ordered_json::object();
  return apply(messages, empty_tools, kwargs);
}

std::optional<std::string> DeepseekV4CppTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  try {
    nlohmann::ordered_json normalized =
        normalize_messages(messages, json_tools);
    normalized = merge_tool_messages(normalized);
    sort_tool_results_by_call_order(&normalized);

    std::string thinking_mode = get_thinking_mode(chat_template_kwargs);
    std::string reasoning_effort = get_reasoning_effort(chat_template_kwargs);
    bool drop_thinking = true;
    if (has_tools(normalized)) {
      drop_thinking = false;
    }
    if (thinking_mode == kThinkingModeThinking && drop_thinking) {
      normalized =
          drop_thinking_messages(normalized, find_last_user_index(normalized));
    }

    std::string prompt;
    prompt +=
        args_.bos_token().empty() ? std::string(kBosToken) : args_.bos_token();
    for (int32_t idx = 0; idx < static_cast<int32_t>(normalized.size());
         ++idx) {
      prompt += render_message(
          normalized, idx, thinking_mode, drop_thinking, reasoning_effort);
    }
    return prompt;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to apply DeepSeek V4 native template: " << e.what();
    return std::nullopt;
  }
}

}  // namespace xllm
