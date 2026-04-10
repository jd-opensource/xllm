/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/util/model_config_utils.h"

#include <glog/logging.h>

#include <algorithm>
#include <filesystem>

namespace xllm::util {

namespace {

bool is_qwen35_multimodal_checkpoint(const JsonReader& reader,
                                     const std::string& model_type) {
  if (model_type != "qwen3_5" || !reader.contains("vision_config")) {
    return false;
  }
  const auto architectures =
      reader.value<std::vector<std::string>>("architectures");
  if (!architectures.has_value()) {
    return true;
  }

  return std::find(architectures->begin(),
                   architectures->end(),
                   "Qwen3_5ForConditionalGeneration") != architectures->end();
}

std::string resolve_model_type(const JsonReader& reader,
                               const std::string& model_type,
                               const std::optional<std::string>& backend) {
  if (!is_qwen35_multimodal_checkpoint(reader, model_type)) {
    return model_type;
  }

  if (backend.has_value() && backend.value() == "vlm") {
    return "qwen3_5_vl";
  }

  const auto text_model_type =
      reader.value<std::string>("text_config.model_type");
  if (text_model_type.has_value()) {
    return text_model_type.value();
  }

  return "qwen3_5_vl";
}

}  // namespace

std::string get_model_type(const JsonReader& reader,
                           const std::filesystem::path& model_path,
                           std::optional<std::string> backend) {
  auto model_type = reader.value<std::string>("model_type");
  if (!model_type.has_value()) {
    model_type = reader.value<std::string>("model_name");
  }
  if (!model_type.has_value()) {
    LOG(FATAL) << "Please check config.json file in model path: " << model_path
               << ", it should contain model_type or model_name key.";
  }

  return resolve_model_type(reader, model_type.value(), backend);
}

std::string get_model_type(const std::filesystem::path& model_path,
                           std::optional<std::string> backend) {
  JsonReader reader;
  // for llm, vlm and rec models, the config.json file is in the model path
  const std::filesystem::path config_json_path = model_path / "config.json";

  if (!std::filesystem::exists(config_json_path)) {
    LOG(FATAL) << "Please check config.json or model_index.json file, one of "
                  "them should exist in the model path: "
               << model_path;
  }
  if (!reader.parse(config_json_path.string())) {
    LOG(FATAL) << "Failed to parse config.json file in model path: "
               << model_path;
  }

  return get_model_type(reader, model_path, std::move(backend));
}

}  // namespace xllm::util
