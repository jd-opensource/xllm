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

#include "multi_layer_xtensor_transfer.h"

#include "common/global_flags.h"

namespace xllm {

void MultiLayerXTensorTransfer::initialize(
    const std::vector<torch::Device>& devices) {
  for (const auto& device : devices) {
    multi_layer_k_xtensor_map_[device.index()] = nullptr;
    multi_layer_v_xtensor_map_[device.index()] = nullptr;
  }
}

void MultiLayerXTensorTransfer::set_multi_layer_xtensor(
    std::vector<std::shared_ptr<XTensor>>& k_xtensors,
    std::vector<std::shared_ptr<XTensor>>& v_xtensors,
    torch::Device device) {
  multi_layer_k_xtensor_map_[device.index()] =
      std::make_unique<MultiLayerXTensor>(k_xtensors);
  multi_layer_v_xtensor_map_[device.index()] =
      std::make_unique<MultiLayerXTensor>(v_xtensors);
}

MultiLayerXTensorPair MultiLayerXTensorTransfer::move_multi_layer_xtensor(
    int32_t device_id) {
  auto k_it = multi_layer_k_xtensor_map_.find(device_id);
  auto v_it = multi_layer_v_xtensor_map_.find(device_id);
  CHECK(k_it != multi_layer_k_xtensor_map_.end())
      << "MultiLayerXTensor not set for device " << device_id;
  CHECK(v_it != multi_layer_v_xtensor_map_.end())
      << "MultiLayerXTensor not set for device " << device_id;

  multi_layer_k_xtensor_map_.erase(k_it);
  multi_layer_v_xtensor_map_.erase(v_it);
  return std::make_pair(std::move(k_it->second), std::move(v_it->second));
}
}  // namespace xllm