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

#include <torch/torch.h>
#if defined(USE_NPU)
#include <torch_npu/torch_npu.h>
#endif

#include <vector>

#include "state_dict.h"

namespace xllm {

namespace weight {

void load_weight(const StateDict& state_dict,
                 const std::string& name,
                 torch::Tensor& weight,
                 bool& weight_is_loaded);

void load_sharded_weight(const StateDict& state_dict,
                         const std::string& name,
                         int64_t dim,
                         int32_t rank,
                         int32_t world_size,
                         torch::Tensor& weight,
                         bool& weight_is_loaded);

using TensorTransform = std::function<torch::Tensor(const torch::Tensor&)>;
void load_sharded_weight(const StateDict& state_dict,
                         const std::string& name,
                         TensorTransform transform_func,
                         int64_t dim,
                         int32_t rank,
                         int32_t world_size,
                         torch::Tensor& weight,
                         bool& weight_is_loaded);

void load_fused_weight(const StateDict& state_dict,
                       const std::vector<std::string>& prefixes,
                       const std::string& name,
                       int64_t dim,
                       int32_t rank,
                       int32_t world_size,
                       std::vector<torch::Tensor>& accumulated_tensors,
                       torch::Tensor& weight,
                       bool& weight_is_loaded);
}  // namespace weight

// helper macros for defining and loading weights
#define DEFINE_WEIGHT(name) \
  torch::Tensor name##_;    \
  bool name##_is_loaded_ = false;

#define DEFINE_FUSED_WEIGHT(name) \
  torch::Tensor name##_;          \
  bool name##_is_loaded_ = false; \
  std::vector<torch::Tensor> name##_list_;

#define LOAD_FUSED_WEIGHT(name, dim)      \
  weight::load_fused_weight(state_dict,   \
                            prefixes,     \
                            #name,        \
                            dim,          \
                            rank,         \
                            world_size,   \
                            name##_list_, \
                            name##_,      \
                            name##_is_loaded_);

#define LOAD_SHARDED_WEIGHT(name, dim) \
  weight::load_sharded_weight(         \
      state_dict, #name, dim, rank, world_size, name##_, name##_is_loaded_);

#define LOAD_SHARDED_WEIGHT_WITH_TRANSFORM(name, dim) \
  weight::load_sharded_weight(state_dict,             \
                              #name,                  \
                              transform_func,         \
                              dim,                    \
                              rank,                   \
                              world_size,             \
                              name##_,                \
                              name##_is_loaded_);

#define LOAD_WEIGHT(name) \
  weight::load_weight(state_dict, #name, name##_, name##_is_loaded_);

}  // namespace xllm
