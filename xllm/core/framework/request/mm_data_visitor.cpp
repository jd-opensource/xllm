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

}  // namespace xllm
