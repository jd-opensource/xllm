/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <memory>
#include <string_view>
#include <unordered_map>

namespace xllm {

class StateDict {
 public:
  StateDict(std::unordered_map<std::string, torch::Tensor> dict,
            const std::string& prefix = "");
  virtual ~StateDict() = default;

  // get the tensor with the given name. return nullptr if not found.
  virtual torch::Tensor get_tensor(const std::string& tensor_name) const;

  // get the sharded tensor with the given name for the given rank.
  virtual torch::Tensor get_sharded_tensor(const std::string& tensor_name,
                                           int64_t dim,
                                           int rank,
                                           int world_size) const;

  // get all the tensors whose name starts with prefix.
  // the returned tensor name will be the suffix of the original name.
  virtual StateDict get_dict_with_prefix(const std::string& prefix) const;

  // get all tensors whose name starts with prefix and apply the transform
  // for each tensor.
  using TensorTransform =
      std::function<torch::Tensor(const std::string&, const torch::Tensor&)>;
  virtual StateDict get_dict_with_prefix(const std::string& prefix,
                                         TensorTransform transform_func) const;

  size_t size() const { return dict_.size(); }

  std::string_view prefix() const { return prefix_; }

  auto begin() const { return dict_.begin(); }
  auto end() const { return dict_.end(); }

 protected:
  std::unordered_map<std::string, torch::Tensor> dict_;

  TensorTransform transform_func_ = nullptr;

  std::string prefix_;
};

struct MemoryMapping {
  void* mapped_addr = nullptr;
  size_t mapped_size = 0;
  int fd = -1;

  MemoryMapping() = default;
};

class StateDictFromSafeTensor : public StateDict {
 public:
  StateDictFromSafeTensor(std::unique_ptr<MemoryMapping> mem_map,
                          std::unordered_map<std::string, torch::Tensor> dict);

  ~StateDictFromSafeTensor();

  static std::unique_ptr<StateDict> load(const std::string& weights_file);

 private:
  // memory mapping for safetensors
  std::unique_ptr<MemoryMapping> mem_map_;
};

}  // namespace xllm
