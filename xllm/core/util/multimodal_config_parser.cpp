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

#include "multimodal_config_parser.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>

namespace xllm {

MMProcessConfig parse_mm_process_config(const std::string& config_json) {
  MMProcessConfig config;
  if (config_json.empty()) {
    return config;
  }

  nlohmann::json json;
  try {
    json = nlohmann::json::parse(config_json);
  } catch (const nlohmann::json::exception& e) {
    LOG(FATAL) << "Invalid --mm_process_config JSON: " << e.what()
               << ", raw value: " << config_json;
  }

  if (!json.is_object()) {
    LOG(FATAL) << "--mm_process_config must be a JSON object.";
  }

  for (auto it = json.begin(); it != json.end(); ++it) {
    if (it.key() != "image") {
      LOG(WARNING) << "Unknown mm_process_config section ignored: " << it.key();
    }
  }

  if (!json.contains("image")) {
    return config;
  }

  const auto& image_cfg = json["image"];
  if (!image_cfg.is_object()) {
    LOG(FATAL) << "--mm_process_config.image must be a JSON object.";
  }

  for (auto it = image_cfg.begin(); it != image_cfg.end(); ++it) {
    if (it.key() != "min_pixels" && it.key() != "max_pixels") {
      LOG(WARNING) << "Unknown mm_process_config.image field ignored: "
                   << it.key();
    }
  }

  if (image_cfg.contains("min_pixels")) {
    if (!image_cfg["min_pixels"].is_number_integer()) {
      LOG(FATAL) << "--mm_process_config.image.min_pixels must be an integer.";
    }
    config.image_min_pixels = image_cfg["min_pixels"].get<int>();
    if (*config.image_min_pixels <= 0) {
      LOG(FATAL) << "--mm_process_config.image.min_pixels must be > 0.";
    }
  }

  if (image_cfg.contains("max_pixels")) {
    if (!image_cfg["max_pixels"].is_number_integer()) {
      LOG(FATAL) << "--mm_process_config.image.max_pixels must be an integer.";
    }
    config.image_max_pixels = image_cfg["max_pixels"].get<int>();
    if (*config.image_max_pixels <= 0) {
      LOG(FATAL) << "--mm_process_config.image.max_pixels must be > 0.";
    }
  }

  return config;
}

}  // namespace xllm
