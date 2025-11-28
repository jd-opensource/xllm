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

#include <absl/strings/match.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/model.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/tensor_util.h"
#include "buffer/atb_workspace.h"
#include "core/layers/base_layer.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/xtensor/xtensor.h"
#include "pytorch/adapter/utils/utils.h"
#include "pytorch/adapter/workspace/workspace.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {
namespace layer {

class NpuBaseLayer : public BaseLayer {
 public:
  explicit NpuBaseLayer(const ModelContext& context);
  ~NpuBaseLayer() override;

  atb::Status execute_node(atb_speed::Model::Node& node,
                           int nodeId = 0,
                           aclrtEvent* event = nullptr,
                           std::atomic<bool>* event_flag = nullptr);

  atb::Status execute_plan(const atb_speed::Model::Node& node,
                           const std::string& op_name,
                           aclrtEvent* event,
                           std::atomic<bool>* event_flag);

  virtual void run_task(std::string taskName,
                        std::function<int()> task) const override;

  void init_weight_slices(int weight_count);

  void copy_weights_to_pinned_host();

  void copy_weights_to_device();

  void copy_weights_to_device_async();

  virtual void init_atb_tensors();

 protected:
  atb::Tensor XTensor2Tensor(const std::shared_ptr<xllm::XTensor>& xtensor);

 protected:
  struct WeightSlice {
    size_t offset = 0;
    size_t bytes = 0;
    std::vector<int64_t> sizes;
    torch::ScalarType dtype = torch::kFloat16;
  };
  void* host_pinned_storage_ = nullptr;
  void* device_storage_ = nullptr;
  size_t storage_size_ = 0;
  std::vector<WeightSlice> weight_slices_;
  atb::Context* context_;
  AtbWorkspace work_space_;
  std::vector<atb::Tensor> atb_weight_tensors_;
  bool graph_captured_{false};
  static constexpr size_t kDeviceAlignment = 64;
  static constexpr size_t kHostAlignment = 64;
  void release_device_storage();
  void release_host_storage();
  torch::Tensor convert_to_torch_tensor(const std::vector<int64_t>& dims,
                                        const torch::ScalarType dtype,
                                        const uintptr_t& dev_addr,
                                        int acl_format = 2);
};

}  // namespace layer
}  // namespace xllm
