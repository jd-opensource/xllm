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

#include <cstring>

#include "core/util/hash_util.h"
#include "core/util/tensor_helper.h"
#include "core/util/utils.h"
#include "mm_data_visitor.h"

namespace xllm {

MMBatchData::MMBatchData(const std::vector<MMData>& datas) {
  this->batch(datas);
}

MMBatchData::MMBatchData(uint32_t type, const MMDict& items)
    : type_(type), data_(std::move(items)) {}

bool MMBatchData::has(const MMKey& key) const {
  if (!valid()) return false;

  const auto& itor = data_.find(key);
  return itor != data_.end();
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

void MMBatchData::to(const torch::Device& device) {
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

  data_ = std::move(dict);
}

MMBatchData MMBatchData::to(const MMBatchData& mm_data,
                            const torch::Device& device) {
  MMBatchData new_mm_data = mm_data;
  new_mm_data.to(device);
  return new_mm_data;
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

  type_ = visitor.type_;
  data_ = std::move(dict);
}

void MMBatchData::debug_print() const {
  LOG(INFO) << "mm batch data debug print, type:" << type_;
  LOG(INFO) << "=============== mm batch vec data ================";
  LOG(INFO) << "mm batch data vec count:" << mm_datas_.size();
  for (const auto& mm_data : mm_datas_) {
    mm_data.debug_print();
  }
  LOG(INFO) << "=============== mm batch data dict data ================";

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

namespace {
bool mmvalue_to_proto(const xllm::MMValue& value, proto::MMValue* pb_value) {
  if (!pb_value) {
    LOG(ERROR) << "PB MMValue pointer is null";
    return false;
  }
  if (std::holds_alternative<torch::Tensor>(value)) {
    auto& torch_tensor = std::get<torch::Tensor>(value);
    proto::Tensor* pb_tensor = pb_value->mutable_single_tensor();
    if (!util::torch_to_proto(torch_tensor, pb_tensor)) {
      LOG(ERROR) << "Failed to convert torch Tensor to PB Tensor";
      return false;
    }
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(value)) {
    auto& torch_tensor_vec = std::get<std::vector<torch::Tensor>>(value);
    proto::TensorList* pb_tensor_list = pb_value->mutable_tensor_list();
    pb_tensor_list->mutable_tensors()->Reserve(torch_tensor_vec.size());
    for (const auto& torch_tensor : torch_tensor_vec) {
      proto::Tensor* pb_tensor = pb_tensor_list->add_tensors();
      if (!util::torch_to_proto(torch_tensor, pb_tensor)) {
        LOG(ERROR) << "Failed to convert torch Tensor to PB Tensor (list item)";
        return false;
      }
    }
  } else {
    LOG(ERROR) << "Unsupported struct MMValue type";
    return false;
  }

  return true;
}

std::optional<xllm::MMValue> proto_to_mmvalue(const proto::MMValue& pb_value) {
  if (pb_value.has_single_tensor()) {
    const auto& pb_tensor = pb_value.single_tensor();
    torch::Tensor torch_tensor = util::proto_to_torch(pb_tensor);
    if (!torch_tensor.defined()) {
      LOG(ERROR) << "Failed to convert PB Tensor to torch Tensor";
      return std::nullopt;
    }
    return xllm::MMValue(torch_tensor);
  } else if (pb_value.has_tensor_list()) {
    const auto& pb_tensor_list = pb_value.tensor_list();
    std::vector<torch::Tensor> torch_tensor_vec;
    torch_tensor_vec.reserve(pb_tensor_list.tensors_size());
    for (const auto& pb_tensor : pb_tensor_list.tensors()) {
      torch::Tensor torch_tensor = util::proto_to_torch(pb_tensor);
      if (!torch_tensor.defined()) {
        LOG(ERROR) << "Failed to convert PB Tensor to torch Tensor (list item)";
        return std::nullopt;
      }
      torch_tensor_vec.emplace_back(std::move(torch_tensor));
    }
    return xllm::MMValue(torch_tensor_vec);
  } else {
    LOG(ERROR) << "PB MMValue has no valid value";
    return std::nullopt;
  }
}

bool mm_item_state_to_proto(const xllm::MMItemState& state,
                            proto::MMItemState* pb_state) {
  if (!pb_state) {
    LOG(ERROR) << "PB MMItemState pointer is null";
    return false;
  }
  auto* pb_token_pos = pb_state->mutable_token_pos();
  pb_token_pos->set_offset(state.token_pos().offset);
  pb_token_pos->set_length(state.token_pos().length);

  auto* pb_prefix_cache = pb_state->mutable_prefix_cache();
  pb_prefix_cache->set_key(state.prefix_cache().key.data,
                           MURMUR_HASH3_VALUE_LEN);
  pb_prefix_cache->set_cached_token_num(state.prefix_cache().cached_token_num);

  return true;
}

std::optional<xllm::MMItemState> proto_to_mm_item_state(
    const proto::MMItemState& pb_state) {
  xllm::MMItemState state;

  if (pb_state.has_token_pos()) {
    state.mutable_token_pos().offset = pb_state.token_pos().offset();
    state.mutable_token_pos().length = pb_state.token_pos().length();
  }

  if (pb_state.has_prefix_cache()) {
    const auto& pb_prefix_cache = pb_state.prefix_cache();
    const auto& key = pb_prefix_cache.key();
    if (!key.empty()) {
      if (key.size() != MURMUR_HASH3_VALUE_LEN) {
        LOG(ERROR) << "PrefixCache key size invalid: " << key.size();
        return std::nullopt;
      }
      std::memcpy(state.mutable_prefix_cache().key.data,
                  key.data(),
                  MURMUR_HASH3_VALUE_LEN);
    } else {
      std::memset(
          state.mutable_prefix_cache().key.data, 0, MURMUR_HASH3_VALUE_LEN);
    }
    state.mutable_prefix_cache().cached_token_num =
        pb_prefix_cache.cached_token_num();
  }

  return state;
}

bool mm_item_to_proto(const xllm::MMDataItem& item,
                      proto::MMDataItem* pb_item) {
  if (!pb_item) {
    LOG(ERROR) << "PB MMDataItem pointer is null";
    return false;
  }
  pb_item->set_type(static_cast<uint32_t>(item.type()));

  auto* pb_dict = pb_item->mutable_dict();
  for (const auto& [key, value] : item.data()) {
    if (key.empty()) {
      LOG(ERROR) << "MMDataItem dict key is empty";
      return false;
    }
    proto::MMValue& pb_value = (*pb_dict)[key];
    if (!mmvalue_to_proto(value, &pb_value)) {
      LOG(ERROR) << "Failed to convert MMDataItem MMValue for key: " << key;
      return false;
    }
  }

  if (!mm_item_state_to_proto(item.state(), pb_item->mutable_state())) {
    LOG(ERROR) << "Failed to convert MMDataItem state";
    return false;
  }

  return true;
}

std::optional<xllm::MMDataItem> proto_to_mm_item(
    const proto::MMDataItem& pb_item) {
  uint32_t type = pb_item.type();
  xllm::MMDict dict;

  for (const auto& [key, pb_value] : pb_item.dict()) {
    if (key.empty()) {
      LOG(ERROR) << "PB MMDataItem dict key is empty";
      return std::nullopt;
    }
    auto value_opt = proto_to_mmvalue(pb_value);
    if (!value_opt) {
      LOG(ERROR) << "Failed to convert PB MMValue for key: " << key;
      return std::nullopt;
    }
    dict.emplace(key, std::move(*value_opt));
  }

  xllm::MMType ty{static_cast<xllm::MMType::Value>(type)};
  xllm::MMDataItem item(ty, dict);

  if (pb_item.has_state()) {
    auto state_opt = proto_to_mm_item_state(pb_item.state());
    if (!state_opt) {
      LOG(ERROR) << "Failed to convert PB MMItemState";
      return std::nullopt;
    }
    item.mutable_state() = std::move(*state_opt);
  }

  return item;
}

bool mmdata_to_proto(const xllm::MMData& mmdata, proto::MMData* pb_mmdata) {
  if (!pb_mmdata) {
    LOG(ERROR) << "PB MMData pointer is null";
    return false;
  }
  if (!mmdata.valid()) {
    pb_mmdata->set_type(xllm::MMType::NONE);
    return true;
  }

  pb_mmdata->set_type(mmdata.type());

  if (mmdata.hold<xllm::MMDict>()) {
    pb_mmdata->set_content_type(proto::MM_DATA_CONTENT_TYPE_DICT);
    auto* pb_dict = pb_mmdata->mutable_dict();
    const auto& dict = mmdata.items<xllm::MMDict>();
    for (const auto& [key, value] : dict) {
      if (key.empty()) {
        LOG(ERROR) << "MMData dict key is empty";
        return false;
      }
      proto::MMValue& pb_value = (*pb_dict)[key];
      if (!mmvalue_to_proto(value, &pb_value)) {
        LOG(ERROR) << "Failed to convert MMData MMValue for key: " << key;
        return false;
      }
    }
  } else if (mmdata.hold<xllm::MMItemVec>()) {
    pb_mmdata->set_content_type(proto::MM_DATA_CONTENT_TYPE_ITEMS);
    const auto& items = mmdata.items<xllm::MMItemVec>();
    pb_mmdata->mutable_items()->Reserve(items.size());
    for (const auto& item : items) {
      if (!mm_item_to_proto(item, pb_mmdata->add_items())) {
        LOG(ERROR) << "Failed to convert MMDataItem";
        return false;
      }
    }
  } else {
    LOG(ERROR) << "Unsupported MMData item type";
    return false;
  }

  return true;
}

std::optional<xllm::MMData> proto_to_mmdata(const proto::MMData& pb_mmdata) {
  uint32_t type = pb_mmdata.type();
  xllm::MMType ty{static_cast<xllm::MMType::Value>(type)};
  proto::MMDataContentType content_type = pb_mmdata.content_type();

  if (content_type == proto::MM_DATA_CONTENT_TYPE_ITEMS) {
    if (!pb_mmdata.dict().empty()) {
      LOG(ERROR) << "PB MMData content=ITEMS but dict is not empty";
      return std::nullopt;
    }
  } else if (content_type == proto::MM_DATA_CONTENT_TYPE_DICT) {
    if (pb_mmdata.items_size() > 0) {
      LOG(ERROR) << "PB MMData content=DICT but items is not empty";
      return std::nullopt;
    }
  }

  if (content_type == proto::MM_DATA_CONTENT_TYPE_ITEMS ||
      (content_type == proto::MM_DATA_CONTENT_TYPE_UNSPECIFIED &&
       pb_mmdata.items_size() > 0)) {
    xllm::MMItemVec vec;
    vec.reserve(pb_mmdata.items_size());
    for (const auto& pb_item : pb_mmdata.items()) {
      auto mm_item = proto_to_mm_item(pb_item);
      if (!mm_item) {
        LOG(ERROR) << "Failed to convert PB MMDataItem";
        return std::nullopt;
      }
      vec.emplace_back(std::move(*mm_item));
    }
    return xllm::MMData(ty, std::move(vec));
  }

  xllm::MMDict dict;
  for (const auto& [key, pb_value] : pb_mmdata.dict()) {
    if (key.empty()) {
      LOG(ERROR) << "PB MMData dict key is empty";
      return std::nullopt;
    }
    auto value_opt = proto_to_mmvalue(pb_value);
    if (!value_opt) {
      LOG(ERROR) << "Failed to convert PB MMValue for key: " << key;
      return std::nullopt;
    }
    dict.emplace(key, std::move(*value_opt));
  }

  if (type == xllm::MMType::NONE && dict.empty()) {
    return xllm::MMData();
  }
  return xllm::MMData(ty, std::move(dict));
}
}  // namespace

bool mmbatchdata_to_proto(const xllm::MMBatchData& mmdata,
                          proto::MMBatchData* pb_mmdata) {
  if (!pb_mmdata) {
    LOG(ERROR) << "PB MMData pointer is null";
    return false;
  }

  pb_mmdata->set_type(mmdata.type());
  auto* pb_dict = pb_mmdata->mutable_dict();

  const auto& dict = mmdata.data();
  for (const auto& [key, value] : dict) {
    if (key.empty()) {
      LOG(ERROR) << "MMData dict key is empty";
      return false;
    }
    proto::MMValue& pb_value = (*pb_dict)[key];
    if (!mmvalue_to_proto(value, &pb_value)) {
      LOG(ERROR) << "Failed to convert struct MMValue for key: " << key;
      return false;
    }
  }

  const auto& mm_datas = mmdata.mm_data_vec();
  if (!mm_datas.empty()) {
    pb_mmdata->mutable_mm_datas()->Reserve(mm_datas.size());
    for (const auto& mm_data : mm_datas) {
      if (!mmdata_to_proto(mm_data, pb_mmdata->add_mm_datas())) {
        LOG(ERROR) << "Failed to convert MMData in mm_datas_";
        return false;
      }
    }
  }

  return true;
}

bool proto_to_mmbatchdata(const proto::MMBatchData& pb_mmdata,
                          xllm::MMBatchData* mmdata) {
  if (!mmdata) {
    LOG(ERROR) << "Struct MMData pointer is null";
    return false;
  }

  uint32_t type = pb_mmdata.type();
  xllm::MMDict dict;

  const auto& pb_dict = pb_mmdata.dict();
  for (const auto& [key, pb_value] : pb_dict) {
    if (key.empty()) {
      LOG(ERROR) << "PB MMData dict key is empty";
      return false;
    }
    auto mm_value = proto_to_mmvalue(pb_value);
    if (!mm_value) {
      LOG(ERROR) << "Failed to convert PB MMValue for key: " << key;
      return false;
    }

    dict.emplace(key, std::move(*mm_value));
  }

  std::vector<xllm::MMData> mm_datas;
  if (pb_mmdata.mm_datas_size() > 0) {
    mm_datas.reserve(pb_mmdata.mm_datas_size());
    for (const auto& pb_mm_data : pb_mmdata.mm_datas()) {
      auto mm_data = proto_to_mmdata(pb_mm_data);
      if (!mm_data) {
        LOG(ERROR) << "Failed to convert PB MMData in mm_datas";
        return false;
      }
      mm_datas.emplace_back(std::move(*mm_data));
    }
  }

  if (!mm_datas.empty()) {
    xllm::MMBatchData batch_mm_data(std::move(mm_datas));
    if (!dict.empty()) {
      batch_mm_data.replace(dict);
    }
    if (type != MMType::NONE && batch_mm_data.type() != type) {
      LOG(WARNING) << "PB MMData type mismatch, use batch type: "
                   << batch_mm_data.type() << " pb type: " << type;
    }
    *mmdata = std::move(batch_mm_data);
  } else {
    *mmdata = xllm::MMBatchData(type, std::move(dict));
  }

  return true;
}

}  // namespace xllm
