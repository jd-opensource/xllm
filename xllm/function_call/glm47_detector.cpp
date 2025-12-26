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

#include "glm47_detector.h"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace xllm {
namespace function_call {

Glm47Detector::Glm47Detector() : BaseFormatDetector() {
  bot_token_ = "<tool_call>";
  eot_token_ = "</tool_call>";

  // Regex patterns for GLM-4.7 format (compact, no newlines)
  // Use [\s\S] instead of . to match any character including newlines
  func_call_regex_ = std::regex("<tool_call>[\\s\\S]*?</tool_call>",
                                std::regex_constants::ECMAScript);
  func_detail_regex_ =
      std::regex("<tool_call>([\\s\\S]*?)(<arg_key>[\\s\\S]*?)?</tool_call>",
                 std::regex_constants::ECMAScript);
  func_arg_regex_ = std::regex(
      "<arg_key>([\\s\\S]*?)</arg_key>(?:\\n|\\s)*<arg_value>([\\s\\S]*?)</"
      "arg_value>",
      std::regex_constants::ECMAScript);

  last_arguments_ = "";
  streamed_raw_length_ = 0;
  reset_streaming_state();
}

void Glm47Detector::reset_streaming_state() {
  stream_state_ = StreamState::INIT;
  current_key_ = "";
  current_value_ = "";
  xml_tag_buffer_ = "";
  is_first_param_ = true;
  value_started_ = false;
  cached_value_type_ = "";
}

std::string Glm47Detector::trim_whitespace(std::string_view str) const {
  const char* whitespace = " \t\n\r";

  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return std::string{};
  }

  size_t end = str.find_last_not_of(whitespace);

  return std::string(str.substr(start, end - start + 1));
}

bool Glm47Detector::has_tool_call(const std::string& text) {
  return text.find(bot_token_) != std::string::npos;
}

std::string Glm47Detector::get_argument_type(
    const std::string& func_name,
    const std::string& arg_key,
    const std::vector<JsonTool>& tools) const {
  // Build name to tool map
  std::unordered_map<std::string, const JsonTool*> name2tool;
  for (const auto& tool : tools) {
    name2tool[tool.function.name] = &tool;
  }

  auto it = name2tool.find(func_name);
  if (it == name2tool.end()) {
    return "";
  }

  const JsonTool* tool = it->second;
  if (!tool->function.parameters.contains("properties")) {
    return "";
  }

  const auto& properties = tool->function.parameters["properties"];
  if (!properties.is_object() || !properties.contains(arg_key)) {
    return "";
  }

  const auto& prop = properties[arg_key];
  if (prop.contains("type") && prop["type"].is_string()) {
    return prop["type"].get<std::string>();
  }

  return "";
}

nlohmann::json Glm47Detector::convert_to_number(
    const std::string& value) const {
  try {
    std::string trimmed = trim_whitespace(value);
    if (trimmed.find('.') != std::string::npos ||
        trimmed.find('e') != std::string::npos ||
        trimmed.find('E') != std::string::npos) {
      return nlohmann::json(std::stod(trimmed));
    } else {
      return nlohmann::json(std::stoll(trimmed));
    }
  } catch (const std::exception&) {
    return nlohmann::json(value);
  }
}

std::pair<nlohmann::json, bool> Glm47Detector::parse_arguments(
    const std::string& json_value,
    const std::string& arg_type) const {
  // Strategy 1: Direct JSON parsing
  try {
    nlohmann::json parsed_value = nlohmann::json::parse(json_value);

    // Type coercion for number type
    if (arg_type == "number" && parsed_value.is_string()) {
      parsed_value = convert_to_number(parsed_value.get<std::string>());
    }

    return {parsed_value, true};
  } catch (const nlohmann::json::parse_error&) {
    // Continue to next strategy
  }

  // Strategy 2: Unescape and parse
  try {
    std::string wrapped = "{\"tmp\": \"" + json_value + "\"}";
    nlohmann::json temp = nlohmann::json::parse(wrapped);
    nlohmann::json parsed_value =
        nlohmann::json::parse(temp["tmp"].get<std::string>());

    if (arg_type == "number" && parsed_value.is_string()) {
      parsed_value = convert_to_number(parsed_value.get<std::string>());
    }

    return {parsed_value, true};
  } catch (const nlohmann::json::parse_error&) {
    // Continue to next strategy
  } catch (const std::exception&) {
    // Continue to next strategy
  }

  // Strategy 3: Treat as string
  try {
    return {nlohmann::json(json_value), true};
  } catch (const std::exception&) {
    return {nlohmann::json(json_value), false};
  }
}

std::unordered_map<std::string, nlohmann::json>
Glm47Detector::parse_argument_pairs(
    const std::vector<std::pair<std::string, std::string>>& pairs,
    const std::string& func_name,
    const std::vector<JsonTool>& tools) const {
  std::unordered_map<std::string, nlohmann::json> arguments;

  for (const auto& [arg_key, arg_value] : pairs) {
    std::string key = trim_whitespace(arg_key);
    std::string value = trim_whitespace(arg_value);

    std::string arg_type = get_argument_type(func_name, key, tools);
    auto [parsed_value, is_good_json] = parse_arguments(value, arg_type);

    if (arg_type == "string") {
      // Only convert to string if explicitly defined as string type
      if (parsed_value.is_string()) {
        arguments[key] = parsed_value;
      } else if (parsed_value.is_object() || parsed_value.is_array()) {
        // If parsed as dict/list but schema says string, convert to JSON string
        arguments[key] = parsed_value.dump();
      } else {
        arguments[key] = parsed_value.dump();
      }
    } else if (arg_type.empty()) {
      // If type is not defined, keep the parsed value as-is
      arguments[key] = is_good_json ? parsed_value : nlohmann::json(value);
    } else {
      // For other types (number, object, array, etc.), use parsed value
      arguments[key] = is_good_json ? parsed_value : nlohmann::json(value);
    }
  }

  return arguments;
}

StreamingParseResult Glm47Detector::detect_and_parse(
    const std::string& text,
    const std::vector<JsonTool>& tools) {
  size_t idx = text.find(bot_token_);
  std::string normal_text =
      (idx != std::string::npos) ? text.substr(0, idx) : text;

  // Trim normal text
  if (!normal_text.empty()) {
    normal_text = trim_whitespace(normal_text);
  }

  if (idx == std::string::npos) {
    return StreamingParseResult(normal_text, {});
  }

  std::vector<ToolCallItem> calls;

  try {
    std::sregex_iterator iter(text.begin(), text.end(), func_call_regex_);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
      std::smatch match = *iter;
      std::string match_result = match.str();

      // Parse function name and arguments
      std::smatch func_detail;
      if (std::regex_search(match_result, func_detail, func_detail_regex_)) {
        std::string func_name = func_detail[1].str();
        std::string func_args = func_detail[2].str();

        // Parse arguments using regex
        std::vector<std::pair<std::string, std::string>> pairs;
        std::sregex_iterator arg_iter(
            func_args.begin(), func_args.end(), func_arg_regex_);
        std::sregex_iterator arg_end;

        for (; arg_iter != arg_end; ++arg_iter) {
          std::smatch arg_match = *arg_iter;
          if (arg_match.size() >= 3) {
            pairs.emplace_back(arg_match[1].str(), arg_match[2].str());
          }
        }

        auto arguments = parse_argument_pairs(pairs, func_name, tools);

        // Create JSON object for parse_base_json
        nlohmann::json match_json;
        match_json["name"] = func_name;
        match_json["parameters"] = arguments;

        auto parsed_calls = parse_base_json(match_json, tools);
        calls.insert(calls.end(), parsed_calls.begin(), parsed_calls.end());
      }
    }

    return StreamingParseResult(normal_text, calls);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in GLM-4.7 detect_and_parse: " << e.what();
    return StreamingParseResult(text, {});
  }
}

std::string Glm47Detector::get_value_type(
    const std::string& func_name,
    const std::string& key,
    const std::vector<JsonTool>& tools) const {
  std::string arg_type = get_argument_type(func_name, key, tools);
  if (!arg_type.empty()) {
    return arg_type;
  }

  // Auto-detect type from value (best effort)
  std::string first_chars = trim_whitespace(current_value_);
  if (first_chars.length() >= 10) {
    first_chars = first_chars.substr(0, 10);
  }

  if (!first_chars.empty()) {
    char first_char = first_chars[0];
    if (std::isdigit(first_char) || first_char == '-' || first_char == '.') {
      return "number";
    } else if (first_char == '{' || first_char == '[') {
      return "object";
    }
  }

  return "string";
}

std::string Glm47Detector::format_value_complete(
    const std::string& value,
    const std::string& value_type) const {
  if (value_type == "string") {
    // Ensure proper JSON string formatting with quotes
    return nlohmann::json(value).dump();
  } else if (value_type == "number") {
    try {
      nlohmann::json num = convert_to_number(trim_whitespace(value));
      return num.dump();
    } catch (const std::exception&) {
      // Fallback to string if not a valid number
      LOG(WARNING) << "Failed to parse '" << value
                   << "' as number, treating as string";
      return nlohmann::json(value).dump();
    }
  } else {
    // For object/array types, return as-is (should already be valid JSON)
    return value;
  }
}

std::string Glm47Detector::process_xml_to_json_streaming(
    const std::string& raw_increment,
    const std::string& func_name,
    const std::vector<JsonTool>& tools) {
  std::string json_output;

  for (char ch : raw_increment) {
    xml_tag_buffer_ += ch;

    if (stream_state_ == StreamState::INIT ||
        stream_state_ == StreamState::BETWEEN) {
      if (xml_tag_buffer_.size() >= 9 &&
          xml_tag_buffer_.substr(xml_tag_buffer_.size() - 9) == "<arg_key>") {
        stream_state_ = StreamState::IN_KEY;
        current_key_ = "";
        xml_tag_buffer_ = "";
        json_output += is_first_param_ ? "{" : ", ";
        is_first_param_ = false;
      }
    } else if (stream_state_ == StreamState::IN_KEY) {
      if (xml_tag_buffer_.size() >= 10 &&
          xml_tag_buffer_.substr(xml_tag_buffer_.size() - 10) == "</arg_key>") {
        current_key_ = xml_tag_buffer_.substr(0, xml_tag_buffer_.size() - 10);
        current_key_ = trim_whitespace(current_key_);
        xml_tag_buffer_ = "";
        stream_state_ = StreamState::WAITING_VALUE;
        json_output += nlohmann::json(current_key_).dump() + ": ";
      }
    } else if (stream_state_ == StreamState::WAITING_VALUE) {
      if (xml_tag_buffer_.size() >= 11 &&
          xml_tag_buffer_.substr(xml_tag_buffer_.size() - 11) ==
              "<arg_value>") {
        stream_state_ = StreamState::IN_VALUE;
        current_value_ = "";
        xml_tag_buffer_ = "";
        value_started_ = false;
        // Determine and cache the value type at the start
        cached_value_type_ = get_value_type(func_name, current_key_, tools);
      }
    } else if (stream_state_ == StreamState::IN_VALUE) {
      if (xml_tag_buffer_.size() >= 12 &&
          xml_tag_buffer_.substr(xml_tag_buffer_.size() - 12) ==
              "</arg_value>") {
        std::string final_value =
            xml_tag_buffer_.substr(0, xml_tag_buffer_.size() - 12);
        current_value_ += final_value;

        // Use cached value type for consistency
        std::string value_type =
            cached_value_type_.empty() ? "string" : cached_value_type_;

        if (value_started_) {
          // Output any remaining content
          if (!final_value.empty()) {
            if (value_type == "string") {
              std::string escaped = nlohmann::json(final_value).dump();
              json_output += escaped.substr(1, escaped.size() - 2);
            } else {
              json_output += final_value;
            }
          }
          // Always output closing quote for string type when value was started
          if (value_type == "string") {
            json_output += "\"";
          }
        } else {
          // Value was never started (empty or complete in one chunk)
          json_output += format_value_complete(current_value_, value_type);
        }

        xml_tag_buffer_ = "";
        stream_state_ = StreamState::BETWEEN;
        current_value_ = "";
        value_started_ = false;
        cached_value_type_ = "";
      } else {
        // Check if buffer could be start of closing tag
        std::string closing_tag = "</arg_value>";
        bool is_potential_closing =
            xml_tag_buffer_.size() <= closing_tag.size() &&
            closing_tag.substr(0, xml_tag_buffer_.size()) == xml_tag_buffer_;

        if (!is_potential_closing) {
          std::string content = xml_tag_buffer_;
          // Use cached value type for consistency
          std::string value_type =
              cached_value_type_.empty() ? "string" : cached_value_type_;

          if (value_type == "string") {
            if (!value_started_) {
              json_output += "\"";
              value_started_ = true;
            }
            if (!content.empty()) {
              std::string escaped = nlohmann::json(content).dump();
              json_output += escaped.substr(1, escaped.size() - 2);
              current_value_ += content;
              xml_tag_buffer_ = "";
            }
          } else if (value_type == "number") {
            if (!content.empty()) {
              if (!value_started_) {
                value_started_ = true;
              }
              json_output += content;
              current_value_ += content;
              xml_tag_buffer_ = "";
            }
          } else {
            // For object/array types, output as-is
            if (!content.empty()) {
              if (!value_started_) {
                value_started_ = true;
              }
              json_output += content;
              current_value_ += content;
              xml_tag_buffer_ = "";
            }
          }
        }
      }
    }
  }

  return json_output;
}

StreamingParseResult Glm47Detector::parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools) {
  buffer_ += new_text;
  std::string current_text = buffer_;

  // Check if we have a tool call
  bool has_tool_call_marker =
      current_text.find(bot_token_) != std::string::npos;

  if (!has_tool_call_marker) {
    // Check if buffer could be the start of a tool call
    bool is_potential_start = false;
    for (size_t i = 1; i <= std::min(current_text.size(), bot_token_.size());
         ++i) {
      if (current_text.size() >= i &&
          bot_token_.substr(0, i) ==
              current_text.substr(current_text.size() - i)) {
        is_potential_start = true;
        break;
      }
    }

    if (!is_potential_start) {
      // Not a potential tool call, return as normal text
      std::string output_text = current_text;
      buffer_.clear();
      // Remove any stray closing tags
      size_t pos = 0;
      while ((pos = output_text.find(eot_token_, pos)) != std::string::npos) {
        output_text.erase(pos, eot_token_.length());
      }
      return StreamingParseResult(output_text, {});
    } else {
      // Could be start of tool call, keep buffering
      return StreamingParseResult("", {});
    }
  }

  // Initialize tool indices if needed
  if (tool_indices_.empty()) {
    tool_indices_ = get_tool_indices(tools);
  }

  std::vector<ToolCallItem> calls;

  try {
    // Try to match a partial or complete tool call
    std::regex partial_regex("<tool_call>(.*?)(<arg_key>.*?)?(</tool_call>|$)",
                             std::regex_constants::ECMAScript);
    std::smatch partial_match;

    if (std::regex_search(current_text, partial_match, partial_regex)) {
      std::string func_name = trim_whitespace(partial_match[1].str());
      std::string func_args_raw = trim_whitespace(partial_match[2].str());
      std::string is_tool_end = partial_match[3].str();

      // Initialize state if this is the first tool call
      if (current_tool_id_ == -1) {
        current_tool_id_ = 0;
        prev_tool_call_arr_.clear();
        streamed_args_for_tool_.clear();
        streamed_args_for_tool_.push_back("");
        streamed_raw_length_ = 0;
        current_tool_name_sent_ = false;
        reset_streaming_state();
      }

      // Ensure we have enough entries in our tracking arrays
      while (prev_tool_call_arr_.size() <=
             static_cast<size_t>(current_tool_id_)) {
        prev_tool_call_arr_.push_back({});
      }
      while (streamed_args_for_tool_.size() <=
             static_cast<size_t>(current_tool_id_)) {
        streamed_args_for_tool_.push_back("");
      }

      // Send tool name first if not sent yet
      if (!current_tool_name_sent_) {
        if (func_name.empty()) {
          LOG(WARNING) << "func_name should not be empty";
        }
        calls.push_back(ToolCallItem(current_tool_id_, func_name, ""));
        current_tool_name_sent_ = true;
        streamed_raw_length_ = 0;
        reset_streaming_state();
        // Store the tool call info
        prev_tool_call_arr_[current_tool_id_]["name"] = func_name;
        prev_tool_call_arr_[current_tool_id_]["arguments"] = "";
      } else {
        // Process XML to JSON streaming
        size_t current_raw_length = func_args_raw.size();

        if (current_raw_length > streamed_raw_length_) {
          // Get the new raw XML content
          std::string raw_increment =
              func_args_raw.substr(streamed_raw_length_);

          // Convert XML increment to JSON increment using state machine
          std::string json_increment =
              process_xml_to_json_streaming(raw_increment, func_name, tools);

          if (!json_increment.empty()) {
            calls.push_back(
                ToolCallItem(current_tool_id_, std::nullopt, json_increment));
            last_arguments_ += json_increment;
            streamed_args_for_tool_[current_tool_id_] += json_increment;
          }

          // Update the streamed length
          streamed_raw_length_ = current_raw_length;
        }

        if (is_tool_end == eot_token_) {
          if (is_first_param_) {
            std::string empty_object = "{}";
            calls.push_back(
                ToolCallItem(current_tool_id_, std::nullopt, empty_object));
            last_arguments_ += empty_object;
          } else if (last_arguments_.empty() || last_arguments_.back() != '}') {
            std::string closing_brace = "}";
            calls.push_back(
                ToolCallItem(current_tool_id_, std::nullopt, closing_brace));
            last_arguments_ += closing_brace;
            streamed_args_for_tool_[current_tool_id_] += closing_brace;
          }

          try {
            std::sregex_iterator arg_iter(
                func_args_raw.begin(), func_args_raw.end(), func_arg_regex_);
            std::sregex_iterator arg_end;

            std::vector<std::pair<std::string, std::string>> pairs;
            for (; arg_iter != arg_end; ++arg_iter) {
              std::smatch arg_match = *arg_iter;
              if (arg_match.size() >= 3) {
                pairs.emplace_back(arg_match[1].str(), arg_match[2].str());
              }
            }

            if (!pairs.empty()) {
              auto arguments = parse_argument_pairs(pairs, func_name, tools);
              nlohmann::json args_json = arguments;
              prev_tool_call_arr_[current_tool_id_]["arguments"] =
                  args_json.dump();
            }
          } catch (const std::exception& e) {
            LOG(WARNING) << "Failed to parse arguments: " << e.what();
          }

          // Remove the completed tool call from buffer
          buffer_ = current_text.substr(partial_match.position(3) +
                                        is_tool_end.length());

          StreamingParseResult result("", calls);
          current_tool_id_++;
          last_arguments_ = "";
          current_tool_name_sent_ = false;
          streamed_raw_length_ = 0;
          reset_streaming_state();
          return result;
        }
      }

      return StreamingParseResult("", calls);
    }

    return StreamingParseResult("", {});

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in parse_streaming_increment: " << e.what();
    return StreamingParseResult(current_text, {});
  }
}

}  // namespace function_call
}  // namespace xllm
