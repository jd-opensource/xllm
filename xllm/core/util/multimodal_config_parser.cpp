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
#include <google/protobuf/util/json_util.h>

#include "multimodal_config.pb.h"

namespace xllm {

MMProcessConfig parse_mm_process_config(const std::string& config_json) {
  MMProcessConfig config;
  if (config_json.empty()) {
    return config;
  }

  proto::MMProcessorConfig proto_config;
  const auto status =
      google::protobuf::util::JsonStringToMessage(config_json, &proto_config);
  if (!status.ok()) {
    LOG(FATAL) << "Invalid --mm_process_config JSON: " << status.ToString()
               << ", raw value: " << config_json;
  }

  if (!proto_config.has_image()) {
    return config;
  }

  const auto& image_cfg = proto_config.image();
  if (image_cfg.has_min_pixels()) {
    config.image_min_pixels = image_cfg.min_pixels();
    if (*config.image_min_pixels <= 0) {
      LOG(FATAL) << "--mm_process_config.image.min_pixels must be > 0.";
    }
  }

  if (image_cfg.has_max_pixels()) {
    config.image_max_pixels = image_cfg.max_pixels();
    if (*config.image_max_pixels <= 0) {
      LOG(FATAL) << "--mm_process_config.image.max_pixels must be > 0.";
    }
  }

  return config;
}

}  // namespace xllm
