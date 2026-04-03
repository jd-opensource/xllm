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

#include <optional>
#include <string>

namespace xllm {

struct MMProcessConfig {
  std::optional<int> image_min_pixels;
  std::optional<int> image_max_pixels;

  std::string to_string() const {
    return "{image_min_pixels=" +
           (image_min_pixels.has_value() ? std::to_string(*image_min_pixels)
                                         : std::string("null")) +
           ", image_max_pixels=" +
           (image_max_pixels.has_value() ? std::to_string(*image_max_pixels)
                                         : std::string("null")) +
           "}";
  }
};

}  // namespace xllm
