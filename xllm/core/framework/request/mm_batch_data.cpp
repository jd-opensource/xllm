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
#include "core/util/utils.h"
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

  ty_ = visitor.ty_;
  data_ = std::move(dict);
}

void MMBatchData::debug_print() const {
  LOG(INFO) << "mm batch data debug print, ty:" << ty_;
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
bool mmvalue_to_proto(const xllm::MMValue& cpp_value,
                      proto::MMValue* pb_value) {
  if (!pb_value) {
    LOG(ERROR) << "PB MMValue pointer is null";
    return false;
  }

  if (std::holds_alternative<torch::Tensor>(cpp_value)) {
    auto& torch_tensor = std::get<torch::Tensor>(cpp_value);
    proto::Tensor* pb_tensor = pb_value->mutable_single_tensor();
    if (!util::torch_to_proto(torch_tensor, pb_tensor)) {
      LOG(ERROR) << "Failed to convert torch Tensor to PB Tensor";
      return false;
    }
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(cpp_value)) {
    auto& torch_tensor_vec = std::get<std::vector<torch::Tensor>>(cpp_value);
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
}  // namespace

bool mmdata_to_proto(const xllm::MMBatchData& cpp_mmdata,
                     proto::MMData* pb_mmdata) {
  if (!pb_mmdata) {
    LOG(ERROR) << "PB MMData pointer is null";
    return false;
  }
  if (!cpp_mmdata.valid()) {
    LOG(ERROR) << "Struct MMData is invalid (type=NONE)";
    return false;
  }

  pb_mmdata->set_type(cpp_mmdata.type());
  auto* pb_dict = pb_mmdata->mutable_dict();

  const auto& cpp_dict = cpp_mmdata.data();
  for (const auto& [key, cpp_value] : cpp_dict) {
    proto::MMValue& pb_value = (*pb_dict)[key];
    if (!mmvalue_to_proto(cpp_value, &pb_value)) {
      LOG(ERROR) << "Failed to convert struct MMValue for key: " << key;
      return false;
    }
  }

  return true;
}

bool proto_to_mmdata(const proto::MMData& pb_mmdata,
                     xllm::MMBatchData* cpp_mmdata) {
  if (!cpp_mmdata) {
    LOG(ERROR) << "Struct MMData pointer is null";
    return false;
  }

  uint32_t type = pb_mmdata.type();
  xllm::MMDict cpp_dict;

  const auto& pb_dict = pb_mmdata.dict();
  for (const auto& [key, pb_value] : pb_dict) {
    auto cpp_value_opt = proto_to_mmvalue(pb_value);
    if (!cpp_value_opt) {
      LOG(ERROR) << "Failed to convert PB MMValue for key: " << key;
      return false;
    }

    cpp_dict.emplace(key, std::move(*cpp_value_opt));
  }
  *cpp_mmdata = xllm::MMBatchData(type, std::move(cpp_dict));

  return true;
}

}  // namespace xllm
