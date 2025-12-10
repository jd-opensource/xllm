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

#include "framework/eplb/expert_buffer_manager.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "xllm_kernels/pytorch/atb_torch/core/include/base_operation.h"
#include "xllm_kernels/pytorch/atb_torch/core/include/graph_operation.h"

namespace xllm {
namespace layer {

class BaseLoader {
 public:
  BaseLoader(uint64_t weight_count, const ModelContext& context);
  virtual ~BaseLoader() = default;

  virtual void load_state_dict(const StateDict& state_dict) {};
  virtual void verify_loaded_weights() const {};
  virtual void verify_loaded_weights(const std::string& prefix) const {};
  virtual void merge_loaded_weights() {};
  virtual void resize_experts_weights(int num_of_device_experts) {};

  torch::Dtype string2dtype(const std::string& dtype_str);

  void correct_tensor_dtype(torch::Tensor& tensor,
                            const std::string& tensorName);

  std::vector<at::Tensor>& get_at_weight_tensors() {
    return at_weight_tensors_;
  }

  std::vector<at::Tensor>& get_at_host_weight_tensors() {
    return at_host_weight_tensors_;
  }

  std::unordered_map<std::string, std::vector<torch::Tensor>>&
  get_experts_weight_tensors() {
    return experts_weights_;
  }

  std::unique_ptr<ExpertBufferManager>& get_expert_shared_buffer() {
    return shared_buffer_;
  }

  std::vector<int32_t>& get_device_expert_list() { return device_expert_list_; }

  atb_torch::TorchTensorMap& get_weights_map() { return weights_map_; }

 protected:
  uint64_t weight_count_;
  xllm::ParallelArgs parallel_args_;
  std::string quantize_type_;
  std::string torch_dtype_;
  torch::ScalarType dtype_;
  torch::TensorOptions options_;
  std::vector<at::Tensor> at_weight_tensors_;
  std::vector<at::Tensor> at_host_weight_tensors_;
  std::unique_ptr<ExpertBufferManager> shared_buffer_ = nullptr;
  std::unordered_map<std::string, torch::Tensor> shared_experts_weights_;
  std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;
  std::vector<int32_t> device_expert_list_;
  atb_torch::TorchTensorMap weights_map_;

  at::Device device_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t dp_rank_;
  int32_t dp_local_tp_rank_;

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  bool to_host = false);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  int dim,
                  bool to_host = false);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  int dim,
                  int rank,
                  int world_size,
                  bool to_host = false);
};

}  // namespace layer
}  // namespace xllm