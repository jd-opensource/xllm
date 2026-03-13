/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "glm4_vision_encoder_loader.h"

namespace xllm {
namespace layer {

enum Glm4VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_POST_NORM_WEIGHT,
  IN_QKV_WEIGHT,
  IN_ATTN_PROJ_WEIGHT,
  IN_LINEAR_GATE_UP_WEIGHT,
  IN_LINEAR_DOWN_WEIGHT,
  IN_LINEAR_UP_WEIGHT,
  IN_LINEAR_GATE_WEIGHT
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_QKV_WEIGHT, "attn.qkv.weight"},
    {IN_ATTN_PROJ_WEIGHT, "attn.proj.weight"},
    {IN_LINEAR_GATE_WEIGHT, "mlp.gate_proj.weight"},
    {IN_LINEAR_UP_WEIGHT, "mlp.up_proj.weight"},
    {IN_LINEAR_DOWN_WEIGHT, "mlp.down_proj.weight"}};

// IN_QKV_WEIGHT is handled in merge_loaded_weights.
static std::map<int, int> WEIGHT_SHARD = {{IN_ATTN_PROJ_WEIGHT, 1},
                                          {IN_LINEAR_UP_WEIGHT, 0},
                                          {IN_LINEAR_GATE_WEIGHT, 0},
                                          {IN_LINEAR_DOWN_WEIGHT, 1}};

Glm4VisionEncoderLoader::Glm4VisionEncoderLoader(uint64_t weight_count,
                                                 const ModelContext& context)
    : BaseLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Glm4VisionEncoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

void Glm4VisionEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Glm4VisionEncoderLoader::merge_loaded_weights() {
  if (parallel_args_.world_size() > 1) {
    get_weights_col_packed_qkv();
  }

  at_weight_tensors_[IN_LINEAR_GATE_UP_WEIGHT] =
      torch::cat({at_weight_tensors_[IN_LINEAR_GATE_WEIGHT],
                  at_weight_tensors_[IN_LINEAR_UP_WEIGHT]},
                 0);
  at_weight_tensors_[IN_LINEAR_GATE_WEIGHT] = torch::empty({}, device_);
  at_weight_tensors_[IN_LINEAR_UP_WEIGHT] = torch::empty({}, device_);
}

void Glm4VisionEncoderLoader::get_weights_col_packed_qkv() {
  int rank = parallel_args_.rank();
  int world_size = parallel_args_.world_size();
  auto qkv_weight = torch::chunk(at_weight_tensors_[IN_QKV_WEIGHT], 3, 0);
  at_weight_tensors_[IN_QKV_WEIGHT] =
      torch::cat({(qkv_weight[0].chunk(world_size, 0))[rank],
                  (qkv_weight[1].chunk(world_size, 0))[rank],
                  (qkv_weight[2].chunk(world_size, 0))[rank]},
                 0);
}

}  // namespace layer
}  // namespace xllm
