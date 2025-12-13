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

#include "mm_batch_data.h"

#include "core/util/tensor_helper.h"
#include "mm_data_visitor.h"

namespace xllm {

MMBatchData::MMBatchData(const std::vector<MMData>& datas) {
  this->batch(datas);
}

MMBatchData::MMBatchData(uint32_t ty, const MMDict& items)
    : ty_(ty), data_(std::move(items)) {}

bool MMBatchData::has(const MMKey& key) const {
  if (!valid()) return false;

  const auto& itor = data_.find(key);
  if (itor != data_.end())
    return true;
  else
    return false;
}

void MMBatchData::get(const MMKey& key, std::vector<torch::Tensor>& vec) const {
  if (!valid()) return;

  const auto& itor = data_.find(key);
  if (itor == data_.end()) return;

  if (std::holds_alternative<torch::Tensor>(itor->second)) {
    vec.push_back(std::get<torch::Tensor>(itor->second));
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(itor->second)) {
    const auto& data = std::get<std::vector<torch::Tensor>>(itor->second);
    vec.insert(vec.end(), data.begin(), data.end());
  }
}

MMBatchData MMBatchData::to(const torch::Device& device) const {
  MMDict dict;

  for (const auto& pair : data_) {
    if (std::holds_alternative<torch::Tensor>(pair.second)) {
      dict[pair.first] =
          safe_to(std::get<torch::Tensor>(pair.second), device, true);
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   pair.second)) {
      const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);

      std::vector<torch::Tensor> vec;
      vec.reserve(lst.size());

      for (const auto& item : lst) {
        vec.emplace_back(safe_to(item, device, true));
      }

      dict[pair.first] = std::move(vec);
    }
  }

  return MMBatchData(ty_, dict);
}

void MMBatchData::batch(const std::vector<MMData>& mm_datas) {
  mm_datas_ = std::move(mm_datas);

  CollectMMDataTensorVisitor visitor;
  this->foreach (static_cast<MMData::IVisitor&>(visitor));

  MMDict dict;
  for (const auto& pair : visitor.datas_) {
    torch::Tensor tar;
    if (safe_concat(pair.second, tar)) {
      dict[pair.first] = tar;
    } else {
      dict[pair.first] = std::move(pair.second);
    }
  }

  ty_ = visitor.ty_;
  data_ = std::move(dict);
}

void MMBatchData::debug_print() const {
  LOG(INFO) << "mm data debug print, ty:" << ty_;

  for (const auto& pair : data_) {
    if (std::holds_alternative<torch::Tensor>(pair.second)) {
      torch::Tensor item = std::get<torch::Tensor>(pair.second);
      LOG(INFO) << " single tensor, key:" << pair.first
                << " device:" << item.device() << " dtype:" << item.dtype()
                << " shape:" << item.sizes();
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   pair.second)) {
      const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);

      for (const auto& item : lst) {
        LOG(INFO) << " vector tensor, key:" << pair.first
                  << " device:" << item.device() << " dtype:" << item.dtype()
                  << " shape:" << item.sizes();
      }
    }
  }
}

}  // namespace xllm
