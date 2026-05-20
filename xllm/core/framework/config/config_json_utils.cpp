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

#include "core/framework/config/config_json_utils.h"

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

DEFINE_string(config_json_file,
              "",
              "Path to a JSON config file. Values in the file override "
              "command-line flag values.");

namespace xllm::config {
namespace {

std::mutex& parsed_json_config_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unique_ptr<std::once_flag>& parsed_json_config_once() {
  static std::unique_ptr<std::once_flag> once_flag =
      std::make_unique<std::once_flag>();
  return once_flag;
}

std::string& parsed_json_config_path() {
  static std::string config_path;
  return config_path;
}

std::optional<JsonReader>& parsed_json_config() {
  static std::optional<JsonReader> json_config;
  return json_config;
}

void load_parsed_json_config() {
  const std::string& config_path = parsed_json_config_path();
  if (config_path.empty()) {
    return;
  }

  JsonReader reader;
  try {
    if (!reader.parse(config_path)) {
      LOG(ERROR) << "Failed to load JSON config file: " << config_path;
      return;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to parse JSON config file: " << config_path
               << ", error: " << e.what();
    return;
  }

  parsed_json_config() = reader;
}

void reset_parsed_json_config_if_path_changed() {
  if (parsed_json_config_path() == FLAGS_config_json_file) {
    return;
  }

  parsed_json_config_path() = FLAGS_config_json_file;
  parsed_json_config().reset();
  parsed_json_config_once() = std::make_unique<std::once_flag>();
}

}  // namespace

JsonReader load_json_file(const std::string& config_path) {
  JsonReader reader;
  if (!config_path.empty()) {
    reader.parse(config_path);
  }
  return reader;
}

JsonReader parse_json_string(std::string_view config_json) {
  JsonReader reader;
  if (!config_json.empty()) {
    reader.parse_text(std::string(config_json));
  }
  return reader;
}

const std::optional<JsonReader>& get_parsed_json_config() {
  std::lock_guard<std::mutex> lock(parsed_json_config_mutex());
  reset_parsed_json_config_if_path_changed();
  std::call_once(*parsed_json_config_once(), load_parsed_json_config);
  return parsed_json_config();
}

}  // namespace xllm::config
