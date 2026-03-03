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

#include "framework/request/utils.h"

#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

namespace xllm {

nlohmann::json proto_struct_to_json(const google::protobuf::Struct& pb_struct) {
  nlohmann::json result = nlohmann::json::object();

  for (const auto& field : pb_struct.fields()) {
    result[field.first] = proto_value_to_json(field.second);
  }

  return result;
}

nlohmann::json proto_value_to_json(const google::protobuf::Value& pb_value) {
  switch (pb_value.kind_case()) {
    case google::protobuf::Value::kNullValue:
      return nlohmann::json(nullptr);

    case google::protobuf::Value::kNumberValue:
      return nlohmann::json(pb_value.number_value());

    case google::protobuf::Value::kStringValue:
      return nlohmann::json(pb_value.string_value());

    case google::protobuf::Value::kBoolValue:
      return nlohmann::json(pb_value.bool_value());

    case google::protobuf::Value::kStructValue:
      return proto_struct_to_json(pb_value.struct_value());

    case google::protobuf::Value::kListValue: {
      nlohmann::json result = nlohmann::json::array();
      for (const auto& item : pb_value.list_value().values()) {
        result.push_back(proto_value_to_json(item));
      }
      return result;
    }

    case google::protobuf::Value::KIND_NOT_SET:
    default:
      return nlohmann::json(nullptr);
  }
}

void copy_json_tools_to_proto(
    const std::vector<xllm::JsonTool>& json_tools,
    google::protobuf::RepeatedPtrField<proto::Tool>* proto_tools) {
  if (json_tools.empty() || proto_tools == nullptr) {
    return;
  }

  proto_tools->Reserve(json_tools.size());
  for (const auto& json_tool : json_tools) {
    auto* proto_tool = proto_tools->Add();
    proto_tool->set_type(json_tool.type.empty() ? "function" : json_tool.type);

    auto* proto_function = proto_tool->mutable_function();
    proto_function->set_name(json_tool.function.name);
    proto_function->set_description(json_tool.function.description);

    if (!json_tool.function.parameters.is_null()) {
      google::protobuf::Struct parameters;
      auto status = google::protobuf::util::JsonStringToMessage(
          json_tool.function.parameters.dump(), &parameters);
      if (status.ok()) {
        *proto_function->mutable_parameters() = std::move(parameters);
      } else {
        LOG(WARNING) << "Failed to serialize tool parameters for tool `"
                     << json_tool.function.name
                     << "` to protobuf: " << status.ToString();
      }
    }
  }
}

std::vector<xllm::JsonTool> parse_tools_from_proto(
    const google::protobuf::RepeatedPtrField<proto::Tool>& proto_tools) {
  std::vector<xllm::JsonTool> tools;
  tools.reserve(proto_tools.size());

  for (const auto& proto_tool : proto_tools) {
    xllm::JsonTool json_tool;
    json_tool.type = proto_tool.type();

    const auto& proto_function = proto_tool.function();
    json_tool.function.name = proto_function.name();
    json_tool.function.description = proto_function.description();

    json_tool.function.parameters =
        proto_function.has_parameters()
            ? proto_struct_to_json(proto_function.parameters())
            : nlohmann::json::object();

    tools.emplace_back(std::move(json_tool));
  }
  return tools;
}

}  // namespace xllm
