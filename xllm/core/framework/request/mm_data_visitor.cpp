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

#include "mm_data_visitor.h"

#include <absl/strings/match.h>

namespace xllm {

bool CollectItemTensorVisitor::visit(MMDataItem& item) {
  for (const auto& pair : item.data()) {
    const auto& key = pair.first;

    if (!black_keys_.empty() && black_keys_.count(key)) {
      continue;
    }

    if (!white_keys_.empty() && !white_keys_.count(key)) {
      continue;
    }

    auto& tar = datas_[pair.first];
    if (std::holds_alternative<torch::Tensor>(pair.second)) {
      tar.emplace_back(std::get<torch::Tensor>(pair.second));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   pair.second)) {
      const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);
      tar.insert(tar.end(), lst.begin(), lst.end());
    }
  }
  return true;
}

bool CollectItemTensorVisitor::visit(const MMKey& key, MMValue& value) {
  if (!black_keys_.empty() && black_keys_.count(key)) {
    return true;
  }

  if (!white_keys_.empty() && !white_keys_.count(key)) {
    return true;
  }

  auto& tar = datas_[key];
  if (std::holds_alternative<torch::Tensor>(value)) {
    tar.push_back(std::get<torch::Tensor>(value));
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
    const auto& lst = std::get<std::vector<torch::Tensor>>(value);
    tar.insert(tar.end(), lst.begin(), lst.end());
  }

  return true;
}

bool CollectMMDataTensorVisitor::visit(MMData& data) {
  ty_ |= data.type();
  data.foreach (item_visitor_);
  return true;
}

bool EncoderInputGatherVisitor::visit(MMDataItem& item) {
  if (item.state().prefix_complete_cached()) return true;

  if (item.is_embedded()) return true;

  for (const auto& [key, value] : item.data()) {
    if (absl::StartsWith(key, filter_prefix_)) continue;

    auto& tar = datas_[key];
    if (std::holds_alternative<torch::Tensor>(value)) {
      tar.push_back(std::get<torch::Tensor>(value));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
      const auto& vec = std::get<std::vector<torch::Tensor>>(value);
      tar.insert(tar.end(), vec.begin(), vec.end());
    }
  }
  return true;
}

bool EncoderInputGatherVisitor::finish(MMBatchData& mm_data) {
  MMDict dict;
  for (const auto& pair : datas_) {
    torch::Tensor tar;
    if (safe_concat(pair.second, tar)) {
      dict[pair.first] = tar;
    } else {
      dict[pair.first] = std::move(pair.second);
    }
  }
  mm_data.replace(dict);
  return true;
}

bool EncoderOutputScatterVisitor::visit(MMDataItem& item) {
  if (item.state().prefix_complete_cached()) return true;

  if (item.is_embedded()) return true;

  std::string prefix;
  int32_t* idx = nullptr;

  if (item.type() == MMType::IMAGE) {
    prefix = "image|";
    idx = &image_idx;
  } else if (item.type() == MMType::VIDEO) {
    prefix = "video|";
    idx = &video_idx;
  } else if (item.type() == MMType::AUDIO) {
    prefix = "audio|";
    idx = &audio_idx;
  } else {
    LOG(FATAL) << " mm data item type invalid, type is " << item.type();
    return true;
  }

  for (const auto& [key, value] : data_) {
    const auto& vec = std::get<std::vector<torch::Tensor>>(value);
    if (absl::StartsWith(key, prefix)) {
      std::string name = key.substr(prefix.length());
      item.add(name, vec[*idx]);
    }
  }
  ++(*idx);
  return true;
}

bool EncoderOutputScatterVisitor::finish() const {
  for (const auto& [key, value] : data_) {
    std::string name = key.substr(0, key.find("|"));
    uint32_t idx = 0;
    if (name == "image") {
      idx = image_idx;
    } else if (name == "video") {
      idx = video_idx;
    } else if (name == "audio") {
      idx = audio_idx;
    } else {
      LOG(FATAL) << "invalid modality key: " << key;
    }
    if (idx != std::get<std::vector<torch::Tensor>>(value).size()) {
      return false;
    }
  }
  return true;
}

bool EncoderEmbeddingGatherVisitor::visit(MMDataItem& item) {
  const auto& state = item.state();
  if (state.prefix_complete_cached()) return true;

  int modality_tokens = state.token_pos().length;
  uint32_t cached_token_num = state.prefix_cache().cached_token_num;
  auto mask = torch::ones({modality_tokens}, torch::dtype(torch::kBool));
  mask.index({torch::indexing::Slice(0, cached_token_num)}) = false;
  for (auto& [key, value] : item.mutable_data()) {
    auto& emb = std::get<torch::Tensor>(value);
    emb = safe_to(emb, device_, true);
    if (absl::StartsWith(key, gather_prefix_)) {
      datas_[key].push_back(emb.index({mask}));
    }
  }
  return true;
}

bool EncoderEmbeddingGatherVisitor::finish(MMBatchData& mm_data) {
  MMDict data;
  torch::Tensor tar;
  for (auto& [key, value] : datas_) {
    if (safe_concat(value, tar)) {
      data[key] = tar;
    } else {
      LOG(ERROR) << "safe concat failed.";
      return false;
    }
  }
  mm_data.replace(data);
  return true;
}

}  // namespace xllm
