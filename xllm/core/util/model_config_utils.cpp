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

#include <filesystem>

#include "core/util/json_reader.h"

namespace xllm {

std::string get_model_type(const std::filesystem::path& model_path) {
  JsonReader reader;
  // for llm, vlm and rec models, the config.json file is in the model path
  std::filesystem::path config_json_path = model_path / "config.json";

  if (std::filesystem::exists(config_json_path)) {
    reader.parse(config_json_path);
    // Prefer model_type (e.g. LLM/VLM); fall back to model_name for configs
    // that only have model_name (e.g. LongCat-Image: {"model_name":
    // "LongCat-Image"}).
    auto model_type = reader.value<std::string>("model_type");
    if (!model_type.has_value()) {
      model_type = reader.value<std::string>("model_name");
    }
    if (!model_type.has_value()) {
      LOG(FATAL) << "Please check config.json file in model path: "
                 << model_path
                 << ", it should contain model_type or model_name key.";
    }
    return model_type.value();
  } else {
    LOG(FATAL) << "Please check config.json or model_index.json file, one of "
                  "them should exist in the model path: "
               << model_path;
  }

  return "";
}

}  // namespace xllm
