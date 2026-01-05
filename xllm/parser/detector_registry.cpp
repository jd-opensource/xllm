/* Copyright 2025 The xLLM Authors. All Rights Reserved.
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

#include "detector_registry.h"

#include <glog/logging.h>

#include "absl/strings/str_join.h"

namespace xllm {

namespace {
#define REGISTER_DETECTOR_DEFAULT_FORCE(name, start, end, default_force) \
  {name, [](bool stream, bool force) {                                   \
     return std::make_unique<ReasoningDetector>(                         \
         start, end, default_force, stream);                             \
   }}

#define REGISTER_DETECTOR(name, start, end)                                 \
  {name, [](bool stream, bool force) {                                      \
     return std::make_unique<ReasoningDetector>(start, end, force, stream); \
   }}
}  // namespace

DetectorRegistry::DetectorRegistry() {
  factories_ = {
      REGISTER_DETECTOR_DEFAULT_FORCE(
          "deepseek-r1", "<think>", "</think>", true),
      REGISTER_DETECTOR("deepseek-v3", "<think>", "</think>"),
      REGISTER_DETECTOR("glm45", "<think>", "</think>"),
      REGISTER_DETECTOR("glm47", "<think>", "</think>"),
      REGISTER_DETECTOR_DEFAULT_FORCE("kimi", "◁think▷", "◁/think▷", false),
      REGISTER_DETECTOR("qwen3", "<think>", "</think>"),
      REGISTER_DETECTOR_DEFAULT_FORCE(
          "qwen3-thinking", "<think>", "</think>", true),
      REGISTER_DETECTOR("step3", "<think>", "</think>"),
  };
}

std::unique_ptr<ReasoningDetector> DetectorRegistry::getDetector(
    const std::string& model_type,
    bool stream_reasoning,
    bool force_reasoning) {
  auto it = factories_.find(model_type);
  if (it == factories_.end()) {
    std::vector<std::string> keys;
    for (const auto& pair : factories_) {
      keys.push_back(pair.first);
    }
    LOG(FATAL) << "Unsupported model type for reasoning parser: " << model_type
               << "\nAvailable model types are: " << absl::StrJoin(keys, ", ");
  }

  return it->second(stream_reasoning, force_reasoning);
};

}  // namespace xllm