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
#include <map>

#include "multi_layer_xtensor.h"
#include "util/type_traits.h"

namespace xllm {

using MultiLayerXTensorPair = std::pair<std::unique_ptr<MultiLayerXTensor>,
                                        std::unique_ptr<MultiLayerXTensor>>;

class MultiLayerXTensorTransfer {
 public:
  static MultiLayerXTensorTransfer& get_instance() {
    static MultiLayerXTensorTransfer instance;
    return instance;
  }

  void initialize(const std::vector<torch::Device>& devices);

  void set_multi_layer_xtensor(
      std::vector<std::shared_ptr<XTensor>>& k_xtensors,
      std::vector<std::shared_ptr<XTensor>>& v_xtensors,
      torch::Device device);

  MultiLayerXTensorPair move_multi_layer_xtensor(int32_t device_id);

 private:
  MultiLayerXTensorTransfer() = default;
  ~MultiLayerXTensorTransfer() = default;
  DISALLOW_COPY_AND_ASSIGN(MultiLayerXTensorTransfer);

 private:
  std::map<int32_t, std::unique_ptr<MultiLayerXTensor>>
      multi_layer_k_xtensor_map_;
  std::map<int32_t, std::unique_ptr<MultiLayerXTensor>>
      multi_layer_v_xtensor_map_;
};

}  // namespace xllm