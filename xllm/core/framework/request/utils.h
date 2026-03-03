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

#pragma once

#include <google/protobuf/repeated_ptr_field.h>
#include <google/protobuf/struct.pb.h>

#include <vector>

#include "chat.pb.h"
#include "core/common/types.h"

namespace xllm {

// Serialize internal JsonTool list into proto::Tool repeated field.
void copy_json_tools_to_proto(
    const std::vector<xllm::JsonTool>& json_tools,
    google::protobuf::RepeatedPtrField<proto::Tool>* proto_tools);

// Parse proto::Tool repeated field into internal JsonTool list.
std::vector<xllm::JsonTool> parse_tools_from_proto(
    const google::protobuf::RepeatedPtrField<proto::Tool>& proto_tools);

// Convert protobuf Struct/Value into nlohmann::json.
nlohmann::json proto_struct_to_json(const google::protobuf::Struct& pb_struct);
nlohmann::json proto_value_to_json(const google::protobuf::Value& pb_value);

}  // namespace xllm
