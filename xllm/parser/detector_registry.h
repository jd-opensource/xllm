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

#pragma once

#include <functional>
#include <memory>

#include "common/macros.h"
#include "parser/reasoning_detector.h"

namespace xllm {
class DetectorRegistry {
 public:
  using DetectorFactory =
      std::function<std::unique_ptr<ReasoningDetector>(bool, bool)>;

  static DetectorRegistry& getInstance() {
    static DetectorRegistry instance;
    return instance;
  }

  std::unique_ptr<ReasoningDetector> getDetector(const std::string& model_type,
                                                 bool stream_reasoning,
                                                 bool force_reasoning);

  bool hasDetector(const std::string& model_type) const {
    return factories_.find(model_type) != factories_.end();
  }

 private:
  DetectorRegistry();

  DISALLOW_COPY_AND_ASSIGN(DetectorRegistry);

  std::unordered_map<std::string, DetectorFactory> factories_;
};

}  // namespace xllm