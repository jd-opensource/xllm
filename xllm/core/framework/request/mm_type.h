/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

namespace xllm {

class MMType {
 public:
  enum Value : uint32_t {
    NONE = 0,
    IMAGE = 1 << 0,
    VIDEO = 1 << 1,
    AUDIO = 1 << 2,
    EMBEDDING = 1 << 3
  };

  MMType() = default;
  MMType(Value v) : value(v) {}
  operator Value() const { return value; }
  explicit operator bool() const = delete;

  bool operator==(MMType rhs) const { return value == rhs.value; }
  bool operator!=(MMType rhs) const { return value != rhs.value; }

  bool operator==(Value v) const { return value == v; }
  bool operator!=(Value v) const { return value != v; }

  std::optional<std::string> to_string();

 private:
  Value value = Value::NONE;
};

struct ImageMetadata {
  int64_t height = 0;
  int64_t width = 0;
};

struct VideoMetadata {
  double fps = 0.0;              // original fps
  int64_t total_num_frames = 0;  // original frames
  double duration = 0.0;
  double sampled_fps = 0.0;
  torch::Tensor frame_indices;
  std::vector<double> timestamps;
};

struct AudioMetadata {
  int64_t sample_rate = 0;
  int64_t num_channels = 0;
  double duration = 0.0;
};

}  // namespace xllm
