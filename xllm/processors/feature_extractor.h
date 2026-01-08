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

#include <torch/torch.h>

#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/request/mm_data.h"
#include "core/framework/request/mm_input.h"
#include "core/util/audio_utils.h"

namespace xllm {

enum class PaddingStrategy { DO_NOT_PAD, LONGEST, MAX_LENGTH };

using BatchFeatureKey = std::string;
using BatchFeatureValue =
    std::variant<torch::Tensor, std::vector<torch::Tensor>>;
using BatchFeatureDict = std::unordered_map<BatchFeatureKey, BatchFeatureValue>;

// TODO: replace the BatchFeature class with mmDict class
struct BatchFeature {
  BatchFeature() = default;
  BatchFeature(const BatchFeatureDict& data) : data_(std::move(data)) {}

  bool has(const BatchFeatureKey& key) const {
    if (data_.empty()) return false;

    const auto& itor = data_.find(key);
    if (itor != data_.end())
      return true;
    else
      return false;
  }

  template <typename T>
  bool update(const BatchFeatureKey& key, const T& value) {
    const auto& itor = data_.find(key);
    if (itor != data_.end()) {
      // Key exists, update it
      data_[key] = value;
      return true;
    } else {
      // Key doesn't exist, add it (same as add method)
      data_.insert({key, value});
      return true;
    }
  }

  template <typename T>
  bool add(const BatchFeatureKey& key, const T& value) {
    const auto& itor = data_.find(key);
    if (itor != data_.end()) return false;
    data_.insert({key, value});
    return true;
  }

  template <typename T>
  std::optional<T> get(const BatchFeatureKey& key) const {
    if (data_.empty()) return std::nullopt;

    const auto& itor = data_.find(key);
    if (itor != data_.end())
      return std::get<T>(itor->second);
    else
      return std::nullopt;
  }

  std::vector<torch::Tensor> get_tensor_vec(const BatchFeatureKey& key) const {
    if (data_.empty()) return {};

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

  const BatchFeatureDict& data() const { return data_; }

  BatchFeatureDict data_;
};

// Audio processor class, keep consistent with the naming of the transformer
class FeatureExtractor {
 public:
  FeatureExtractor(int64_t feature_size,
                   int64_t sampling_rate,
                   std::optional<double> padding_value = std::nullopt,
                   const std::string& padding_side = "right",
                   bool return_attention_mask = true);

  virtual ~FeatureExtractor() = default;

  virtual bool process(const MMInput& mm_inputs, MMData& mm_datas) = 0;

  int64_t get_feature_size() const { return feature_size_; }

  int64_t get_sampling_rate() const { return sampling_rate_; }

  void pad(BatchFeature& processed_features,
           PaddingStrategy padding = PaddingStrategy::LONGEST,
           std::optional<int64_t> max_length = std::nullopt,
           bool truncation = false,
           std::optional<int64_t> pad_to_multiple_of = std::nullopt,
           std::optional<bool> return_attention_mask = std::nullopt);

 protected:
  void pad_(BatchFeature& processed_features,
            PaddingStrategy padding_strategy = PaddingStrategy::DO_NOT_PAD,
            std::optional<int64_t> max_length = std::nullopt,
            std::optional<int64_t> pad_to_multiple_of = std::nullopt,
            std::optional<bool> return_attention_mask = std::nullopt);

  void truncate_(BatchFeature& processed_features,
                 std::optional<int64_t> max_length = std::nullopt,
                 std::optional<int64_t> pad_to_multiple_of = std::nullopt,
                 std::optional<bool> truncation = std::nullopt);

  void check_padding_strategy_(
      PaddingStrategy padding,
      std::optional<int64_t> max_length = std::nullopt);

 protected:
  int64_t feature_size_;
  int64_t sampling_rate_;
  std::optional<double> padding_value_;
  std::string padding_side_;
  bool return_attention_mask_;
  std::vector<std::string> model_input_names_;
};

}  // namespace xllm
