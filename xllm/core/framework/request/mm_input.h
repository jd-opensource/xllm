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
#include "core/framework/request/request.h"
#include "mm_type.h"

namespace xllm {

struct MMInputItem {
  void clear() {
    type = MMType::NONE;
    raw_data.clear();
  }

  MMType type = MMType::NONE;
  std::string raw_data;       // binary
  torch::Tensor decode_data;  // image: rgb, [c,h,w], uint8
  VideoMetadata video_meta;
  EmbeddingOutput embedding;
};

struct MMPayload {
  MMPayload() = default;
  explicit MMPayload(const std::string& data, size_t offset = 0)
      : data(std::move(data)), offset(offset) {}

  bool get(std::string& value, size_t len) {
    if (len == data.size()) {
      value = std::move(data);
      return true;
    }

    if (data.size() - offset < len) {
      return false;
    }

    value = data.substr(offset, len);
    offset += len;

    return true;
  }

  std::string data;
  size_t offset;
};

class MMInput {
 public:
  MMInput() = default;
  explicit MMInput(const std::string& payload) : payload(std::move(payload)) {}

  bool empty() const { return items_.empty(); }
  void clear() { items_.clear(); }
  size_t size() const { return items_.size(); }

  const std::vector<MMInputItem>& items() const { return items_; }

  std::vector<MMInputItem>::iterator begin() { return items_.begin(); }

  std::vector<MMInputItem>::iterator end() { return items_.end(); }

  std::vector<MMInputItem>::const_iterator begin() const {
    return items_.begin();
  }

  std::vector<MMInputItem>::const_iterator end() const { return items_.end(); }

  void insert(const std::vector<MMInputItem>& inputs) {
    items_.insert(items_.end(), inputs.begin(), inputs.end());
  }

  std::vector<torch::Tensor> get_decode_data(MMType type) const {
    std::vector<torch::Tensor> vec;

    for (const auto& item : items_) {
      if (item.type == type) {
        vec.emplace_back(item.decode_data);
      }
    }
    return std::move(vec);
  }

  std::vector<VideoMetadata> get_video_metadata() const {
    std::vector<VideoMetadata> metas;
    metas.reserve(items_.size());
    for (auto& item : items_) {
      if (item.type == MMType::VIDEO) {
        metas.push_back(item.video_meta);
      }
    }
    return metas;
  }

  MMPayload payload;

 private:
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
