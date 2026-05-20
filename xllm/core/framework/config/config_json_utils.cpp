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

DEFINE_string(config_json_file,
              "",
              "Path to a JSON config file. Values in the file override "
              "command-line flag values.");

namespace xllm::config {

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

}  // namespace xllm::config
