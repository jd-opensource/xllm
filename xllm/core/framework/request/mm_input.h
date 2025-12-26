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

#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "mm_type.h"

namespace xllm {

struct MMInputItem {
  void clear() {
    type_ = MMType::NONE;
    raw_data_.clear();
  }

  MMType type_ = MMType::NONE;

  std::string raw_data_;  // binary

  torch::Tensor decode_data_;  // image: rgb, [c,h,w], uint8

  VideoMetadata video_meta_;
};

struct MMPayload {
  MMPayload() = default;
  explicit MMPayload(const std::string& data, size_t offset = 0)
      : data_(std::move(data)), offset_(offset) {}

  bool get(std::string& value, size_t len) {
    if (len == data_.size()) {
      value = std::move(data_);
      return true;
    }

    if (data_.size() - offset_ < len) {
      return false;
    }

    value = data_.substr(offset_, len);
    offset_ += len;

    return true;
  }

  std::string data_;
  size_t offset_;
};

struct MMInput {
  MMInput() = default;
  explicit MMInput(const std::string& payload) : payload_(payload) {}

  bool empty() const { return items_.empty(); }
  void clear() { items_.clear(); }

  void insert(const std::vector<MMInputItem>& items) {
    items_.insert(items_.end(), items.begin(), items.end());
  }

  std::vector<torch::Tensor> get_decode_data(MMType type) const {
    std::vector<torch::Tensor> vec;

    for (const auto& item : items_) {
      if (item.type_ == type) {
        vec.emplace_back(item.decode_data_);
      }
    }
    return std::move(vec);
  }

  std::vector<VideoMetadata> get_video_metadata() const {
    std::vector<VideoMetadata> metas;
    metas.reserve(items_.size());
    for (auto& item : items_) {
      if (item.type_ == MMType::VIDEO) {
        metas.push_back(item.video_meta_);
      }
    }
    return metas;
  }

  MMPayload payload_;
  std::vector<MMInputItem> items_;
};

class MMHandlerSet;
class MMInputTransfer {
 public:
  MMInputTransfer();
  ~MMInputTransfer();

  bool trans(const std::vector<Message>& messages, MMInput& inputs);

 private:
  bool trans(const MMContentVec& mmc,
             std::vector<MMInputItem>& inputs,
             MMPayload& payload);

  std::unique_ptr<MMHandlerSet> mm_handlers_;
};

}  // namespace xllm
