#include "tools_converter.h"

#include <glog/logging.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>

#include <algorithm>
#include <random>
#include <regex>
#include <sstream>

namespace llm {

std::string ToolsConverter::convert_tools_to_json(const std::vector<Tool>& tools) {
  if (tools.empty()) {
    return "[]";
  }

  nlohmann::json tools_json = nlohmann::json::array();
  
  for (const auto& tool : tools) {
    nlohmann::json tool_json;
    tool_json["type"] = tool.type;
    
    nlohmann::json function_json;
    function_json["name"] = tool.function.name;
    function_json["description"] = tool.function.description;
    
    try {
      if (!tool.function.parameters.empty()) {
        function_json["parameters"] = nlohmann::json::parse(tool.function.parameters);
      } else {
        function_json["parameters"] = nlohmann::json::object();
      }
    } catch (const nlohmann::json::exception& e) {
      LOG(WARNING) << "Failed to parse tool parameters JSON: " << e.what()
                   << ", tool: " << tool.function.name;
      function_json["parameters"] = nlohmann::json::object();
    }
    
    tool_json["function"] = function_json;
    tools_json.push_back(tool_json);
  }
  
  return tools_json.dump(2);
}

std::string ToolsConverter::convert_tools_to_prompt(
    const std::vector<Tool>& tools,
    const std::string& tool_choice) {
  if (tools.empty()) {
    return "";
  }

  std::ostringstream prompt;
  prompt << "You have access to the following functions:\n\n";
  
  for (const auto& tool : tools) {
    prompt << "Function: " << tool.function.name << "\n";
    prompt << "Description: " << tool.function.description << "\n";
    
    try {
      if (!tool.function.parameters.empty()) {
        auto params_json = nlohmann::json::parse(tool.function.parameters);
        prompt << "Parameters: " << params_json.dump(2) << "\n";
      }
    } catch (const nlohmann::json::exception& e) {
      LOG(WARNING) << "Failed to parse parameters for tool: " << tool.function.name;
    }
    
    prompt << "\n";
  }
  
  if (tool_choice == "required") {
    prompt << "You MUST call one of the above functions. ";
  } else if (tool_choice == "auto") {
    prompt << "You may call one of the above functions if needed. ";
  }
  
  prompt << "To call a function, respond with a JSON object in the following format:\n";
  prompt << "{\n";
  prompt << "  \"tool_calls\": [\n";
  prompt << "    {\n";
  prompt << "      \"id\": \"call_<unique_id>\",\n";
  prompt << "      \"type\": \"function\",\n";
  prompt << "      \"function\": {\n";
  prompt << "        \"name\": \"function_name\",\n";
  prompt << "        \"arguments\": \"{\\\"param1\\\": \\\"value1\\\"}\"\n";
  prompt << "      }\n";
  prompt << "    }\n";
  prompt << "  ]\n";
  prompt << "}\n\n";
  
  return prompt.str();
}

std::vector<ToolCall> ToolsConverter::parse_tool_calls_from_text(
    const std::string& model_output) {
  std::vector<ToolCall> tool_calls;
  
  auto json_blocks = extract_json_blocks(model_output);
  
  for (const auto& json_block : json_blocks) {
    auto parsed_calls = parse_tool_calls_from_json(json_block);
    tool_calls.insert(tool_calls.end(), parsed_calls.begin(), parsed_calls.end());
  }
  
  return tool_calls;
}

std::vector<ToolCall> ToolsConverter::parse_tool_calls_from_json(
    const std::string& json_str) {
  std::vector<ToolCall> tool_calls;
  
  try {
    auto json_obj = nlohmann::json::parse(clean_json_string(json_str));
    
    if (json_obj.contains("tool_calls") && json_obj["tool_calls"].is_array()) {
      for (const auto& call_json : json_obj["tool_calls"]) {
        auto tool_call = parse_single_function_call(call_json);
        if (tool_call.has_value()) {
          tool_calls.push_back(tool_call.value());
        }
      }
    }
    else if (json_obj.contains("function") || json_obj.contains("name")) {
      auto tool_call = parse_single_function_call(json_obj);
      if (tool_call.has_value()) {
        tool_calls.push_back(tool_call.value());
      }
    }
  } catch (const nlohmann::json::exception& e) {
    LOG(WARNING) << "Failed to parse tool calls JSON: " << e.what();
  }
  
  return tool_calls;
}

bool ToolsConverter::validate_tool_call_arguments(
    const ToolCall& tool_call,
    const std::vector<Tool>& available_tools) {
  auto tool_it = std::find_if(available_tools.begin(), available_tools.end(),
                              [&](const Tool& tool) {
                                return tool.function.name == tool_call.function_name;
                              });
  
  if (tool_it == available_tools.end()) {
    LOG(WARNING) << "Tool not found: " << tool_call.function_name;
    return false;
  }
  
  try {
    nlohmann::json::parse(tool_call.function_arguments);
  } catch (const nlohmann::json::exception& e) {
    LOG(WARNING) << "Invalid arguments JSON for tool " << tool_call.function_name
                 << ": " << e.what();
    return false;
  }
  
  return validate_json_schema(tool_call.function_arguments, tool_it->function.parameters);
}

std::string ToolsConverter::generate_tool_call_id() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(100000, 999999);
  
  return "call_" + std::to_string(dis(gen));
}

std::string ToolsConverter::format_tool_choice(const std::string& tool_choice) {
  if (tool_choice == "auto" || tool_choice == "none" || tool_choice == "required") {
    return tool_choice;
  }
  return "auto";


std::optional<ToolCall> ToolsConverter::parse_single_function_call(
    const nlohmann::json& json_obj) {
  try {
    ToolCall tool_call;
    
    if (json_obj.contains("id")) {
      tool_call.id = json_obj["id"].get<std::string>();
    } else {
      tool_call.id = generate_tool_call_id();
    }
    
    if (json_obj.contains("type")) {
      tool_call.type = json_obj["type"].get<std::string>();
    } else {
      tool_call.type = "function";
    }
    
    if (json_obj.contains("function")) {
      const auto& func_json = json_obj["function"];
      if (func_json.contains("name")) {
        tool_call.function_name = func_json["name"].get<std::string>();
      }
      if (func_json.contains("arguments")) {
        if (func_json["arguments"].is_string()) {
          tool_call.function_arguments = func_json["arguments"].get<std::string>();
        } else {
          tool_call.function_arguments = func_json["arguments"].dump();
        }
      }
    }
    else if (json_obj.contains("name")) {
      tool_call.function_name = json_obj["name"].get<std::string>();
      if (json_obj.contains("arguments")) {
        if (json_obj["arguments"].is_string()) {
          tool_call.function_arguments = json_obj["arguments"].get<std::string>();
        } else {
          tool_call.function_arguments = json_obj["arguments"].dump();
        }
      }
    }
    
    if (!tool_call.function_name.empty()) {
      return tool_call;
    }
  } catch (const nlohmann::json::exception& e) {
    LOG(WARNING) << "Failed to parse single function call: " << e.what();
  }
  
  return std::nullopt;
}

bool ToolsConverter::validate_json_schema(
    const std::string& json_str,
    const std::string& schema_str) {
  try {
    nlohmann::json::parse(json_str);
    if (!schema_str.empty()) {
      nlohmann::json::parse(schema_str);
    }
    return true;
  } catch (const nlohmann::json::exception& e) {
    return false;
  }
  

}

std::string ToolsConverter::clean_json_string(const std::string& raw_json) {
  std::string cleaned = raw_json;
  
  cleaned = std::string(absl::StripAsciiWhitespace(cleaned));
  
  cleaned = absl::StrReplaceAll(cleaned, {{"```json", ""}, {"```", ""}});
  
  return cleaned;
}

std::vector<std::string> ToolsConverter::extract_json_blocks(const std::string& text) {
  std::vector<std::string> json_blocks;
  
  std::regex json_regex(R"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})");
  std::sregex_iterator iter(text.begin(), text.end(), json_regex);
  std::sregex_iterator end;
  
  for (; iter != end; ++iter) {
    std::string match = iter->str();
    try {
      nlohmann::json::parse(match);
      json_blocks.push_back(match);
    } catch (const nlohmann::json::exception&) {
      continue;
    }
  }
  
  if (json_blocks.empty()) {
    try {
      nlohmann::json::parse(clean_json_string(text));
      json_blocks.push_back(clean_json_string(text));
    } catch (const nlohmann::json::exception&) {
    }
  }
  
  return json_blocks;
}

}  // namespace llm