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

#include "glm4_vision_encoder_manual_loader.h"

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

static std::map<int, int> WEIGHT_SHARD = {{IN_ATTN_PROJ_WEIGHT, 1},
                                          {IN_LINEAR_UP_WEIGHT, 0},
                                          {IN_LINEAR_GATE_WEIGHT, 0},
                                          {IN_LINEAR_DOWN_WEIGHT, 1}};

Glm4VisionEncoderManualLoader::Glm4VisionEncoderManualLoader(
    uint64_t weight_count,
    const ModelContext& context)
    : BaseManualLoader(weight_count, context) {
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  auto host_options =
      torch::TensorOptions().dtype(options.dtype()).device(torch::kCPU);

  encode_param_rank_ = parallel_args.rank();
  encode_param_world_size_ = parallel_args.world_size();
  at_host_weight_tensors_.resize(weight_count);
  for (int i = 0; i < weight_count; ++i) {
    at_host_weight_tensors_[i] = torch::zeros({1}, host_options);
  }
}

void Glm4VisionEncoderManualLoader::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], true);
    } else {
      set_weight(state_dict, name, index, true);
    }
  }
}

void Glm4VisionEncoderManualLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_host_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Glm4VisionEncoderManualLoader::merge_host_at_weights() {
  auto make_zero_like = [](const torch::Tensor& ref) {
    return torch::zeros(
        {1},
        torch::TensorOptions().dtype(ref.scalar_type()).device(torch::kCPU));
  };

  if (encode_param_world_size_ > 1) {
    get_weights_col_packed_qkv();
  }

  at_host_weight_tensors_[IN_LINEAR_GATE_UP_WEIGHT] =
      torch::cat({at_host_weight_tensors_[IN_LINEAR_GATE_WEIGHT],
                  at_host_weight_tensors_[IN_LINEAR_UP_WEIGHT]},
                 0);

  at_host_weight_tensors_[IN_LINEAR_GATE_WEIGHT] =
      make_zero_like(at_host_weight_tensors_[IN_LINEAR_GATE_WEIGHT]);
  at_host_weight_tensors_[IN_LINEAR_UP_WEIGHT] =
      make_zero_like(at_host_weight_tensors_[IN_LINEAR_UP_WEIGHT]);
}

void Glm4VisionEncoderManualLoader::get_weights_col_packed_qkv() {
  auto qkv_weight = torch::chunk(at_host_weight_tensors_[IN_QKV_WEIGHT], 3, 0);
  at_host_weight_tensors_[IN_QKV_WEIGHT] = torch::cat(
      {qkv_weight[0].chunk(encode_param_world_size_, 0)[encode_param_rank_],
       qkv_weight[1].chunk(encode_param_world_size_, 0)[encode_param_rank_],
       qkv_weight[2].chunk(encode_param_world_size_, 0)[encode_param_rank_]},
      0);
}

}  // namespace layer
}  // namespace xllm
