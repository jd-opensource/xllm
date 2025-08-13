#include "base_format_detector.h"

#include <iostream>
#include <regex>
#include <sstream>

namespace llm {
namespace function_call {

BaseFormatDetector::BaseFormatDetector()
    : current_tool_id_(-1),
      current_tool_name_sent_(false),
      bot_token_(""),
      eot_token_(""),
      tool_call_separator_(", ") {}

std::unordered_map<std::string, int> BaseFormatDetector::get_tool_indices(
    const std::vector<JsonTool>& tools) {
  std::unordered_map<std::string, int> indices;
  for (size_t i = 0; i < tools.size(); ++i) {
    if (!tools[i].function.name.empty()) {
      indices[tools[i].function.name] = static_cast<int>(i);
    } else {
      LOG(ERROR) << "Tool at index " << i
                 << " has empty function name, skipping";
    }
  }
  return indices;
}

std::vector<ToolCallItem> BaseFormatDetector::parse_base_json(
    const nlohmann::json& json_obj,
    const std::vector<JsonTool>& tools) {
  auto tool_indices = get_tool_indices(tools);
  std::vector<ToolCallItem> results;

  std::vector<nlohmann::json> actions;
  if (json_obj.is_array()) {
    for (const auto& item : json_obj) {
      actions.emplace_back(item);
    }
  } else {
    actions.emplace_back(json_obj);
  }

  for (const auto& act : actions) {
    if (!act.is_object()) {
      LOG(ERROR) << "Invalid tool call item, expected object, got: "
                 << act.type_name();
      continue;
    }

    std::string name;
    if (act.contains("name") && act["name"].is_string()) {
      name = act["name"].get<std::string>();
    } else {
      LOG(ERROR) << "Invalid tool call: missing 'name' field or invalid type";
      continue;
    }

    if (tool_indices.find(name) == tool_indices.end()) {
      LOG(ERROR) << "Model attempted to call undefined function: " << name;
      continue;
    }

    nlohmann::json parameters = nlohmann::json::object();

    if (act.contains("parameters")) {
      parameters = act["parameters"];
    } else if (act.contains("arguments")) {
      parameters = act["arguments"];
    } else {
      LOG(ERROR) << "No parameters or arguments field found for tool: " << name;
    }

    if (!parameters.is_object()) {
      LOG(ERROR) << "Invalid arguments type for tool: " << name
                 << ", expected object, got: " << parameters.type_name();
      parameters = nlohmann::json::object();
    }

    std::string parameters_str;
    try {
      parameters_str = parameters.dump(
          -1, ' ', false, nlohmann::json::error_handler_t::ignore);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to serialize arguments for tool: " << name
                 << ", error: " << e.what();
      parameters_str = "{}";
    }

    results.emplace_back(-1, name, parameters_str);
  }

  return results;
}

}  // namespace function_call
}  // namespace llm