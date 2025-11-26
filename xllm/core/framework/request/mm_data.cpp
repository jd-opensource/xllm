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

#include "mm_data.h"

#include "core/util/tensor_helper.h"

namespace xllm {

std::optional<std::string> MMType::to_string() {
  switch (value) {
    case Value::NONE:
      return std::nullopt;
    case Value::IMAGE:
      return "image";
    case Value::VIDEO:
      return "video";
    case Value::AUDIO:
      return "audio";
    default:
      LOG(WARNING) << "Unknown mm type: " << static_cast<int>(value);
  }
  return std::nullopt;
}

void MMData::debug_print() const {
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
    } else {
      assert(0);
    }
  }
}

MMData MMData::to(const MMData& mm_data, const torch::Device& device) {
  MMDict dict;

  for (const auto& pair : mm_data.data()) {
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

      dict[pair.first] = vec;
    } else {
      assert(0);
    }
  }

  return std::move(MMData(mm_data.type(), dict));
}

MMData MMData::batch(const std::vector<MMData>& mm_datas) {
  uint32_t ty = 0;
  std::unordered_map<MMKey, std::vector<torch::Tensor>> lists;

  for (const auto& item : mm_datas) {
    ty |= item.type();
    for (const auto& pair : item.data()) {
      if (std::holds_alternative<torch::Tensor>(pair.second)) {
        lists[pair.first].emplace_back(std::get<torch::Tensor>(pair.second));
      } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                     pair.second)) {
        auto& tar = lists[pair.first];
        const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);
        tar.insert(tar.end(), lst.begin(), lst.end());
      } else {
        assert(0);
      }
    }
  }

  auto check = [](const std::vector<torch::Tensor>& vec) {
    if (vec.empty()) return false;

    const int64_t ref_dim = vec[0].dim();
    if (ref_dim == 0) return false;

    const auto ref_shape = vec[0].sizes().slice(1);
    for (size_t i = 1; i < vec.size(); ++i) {
      if (vec[i].dim() != ref_dim || vec[i].sizes().slice(1) != ref_shape) {
        return false;
      }
    }
    return true;
  };

  MMDict dict;
  for (const auto& pair : lists) {
    if (check(pair.second)) {
      dict[pair.first] = torch::cat(pair.second);
    } else {
      dict[pair.first] = std::move(pair.second);
    }
  }

  return std::move(MMData(ty, dict));
}

}  // namespace xllm
