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
#include <unordered_map>
#include <variant>
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

struct VideoMetadata {
  double fps = 0.0;              // original fps
  int64_t total_num_frames = 0;  // original frames
  double duration = 0.0;
  double sampled_fps = 0.0;
  torch::Tensor frame_indices;
  std::vector<double> timestamps;
};

using MMImageNumberType = int64_t;
using MMKey = std::string;
using MMValue =
    std::variant<torch::Tensor, std::vector<torch::Tensor>, MMImageNumberType>;
using MMDict = std::unordered_map<MMKey, MMValue>;

struct MMData {
  MMData() = default;
  MMData(uint32_t ty, const MMDict& data) : ty_(ty), data_(std::move(data)) {}

  bool has(uint32_t type) const { return type & ty_ != 0; }
  bool has(const MMKey& key) const {
    if (!valid()) return false;

    const auto& itor = data_.find(key);
    if (itor != data_.end())
      return true;
    else
      return false;
  }

  template <typename T>
  bool add(uint32_t type, const MMKey& key, const T& value) {
    const auto& itor = data_.find(key);
    if (itor != data_.end()) return false;

    ty_ |= type;
    data_.insert({key, value});
    return true;
  }

  template <typename T>
  bool update(uint32_t type, const MMKey& key, const T& value) {
    const auto& itor = data_.find(key);
    if (itor != data_.end()) {
      // Key exists, update it
      data_[key] = value;
      ty_ |= type;
      return true;
    } else {
      // Key doesn't exist, add it (same as add method)
      ty_ |= type;
      data_.insert({key, value});
      return true;
    }
  }

  template <typename T>
  std::optional<T> get(const MMKey& key) const {
    if (!valid()) return std::nullopt;

    const auto& itor = data_.find(key);
    if (itor != data_.end())
      return std::get<T>(itor->second);
    else
      return std::nullopt;
  }

  std::vector<torch::Tensor> get_tensor_vec(const MMKey& key) const {
    if (!valid()) return {};

    const auto& itor = data_.find(key);
    if (itor == data_.end()) return {};

    if (std::holds_alternative<torch::Tensor>(itor->second)) {
      return {std::get<torch::Tensor>(itor->second)};
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   itor->second)) {
      return std::get<std::vector<torch::Tensor>>(itor->second);
    } else {
      assert(0);
      return {};
    }
  }

  bool valid() const { return ty_ != MMType::NONE; }

  uint32_t type() const { return ty_; }

  const MMDict& data() const { return data_; }

  void debug_print() const;

  const std::vector<VideoMetadata>& get_video_metadata() const {
    return video_metadata_;
  }

  void set_video_metadata(const std::vector<VideoMetadata>& meta) {
    video_metadata_ = meta;
  }

  static MMData to(const MMData& mm_data, const torch::Device& device);
  static MMData batch(const std::vector<MMData>& mm_datas);

  uint32_t ty_ = MMType::NONE;
  MMDict data_;

 private:
  std::vector<VideoMetadata> video_metadata_;
};

}  // namespace xllm
