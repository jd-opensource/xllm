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

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include "qwen2_vision_encoder_loader.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

enum VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS,
  IN_POST_NORM_WEIGHT,
  IN_POST_NORM_BIAS,
  IN_QKV_WEIGHT,
  IN_QKV_BIAS,
  IN_WATTENTION_OUT_WEIGHT,
  IN_WATTENTION_OUT_BIAS,
  IN_LINEAR_FC1_WEIGHT,
  IN_LINEAR_FC1_BIAS,
  IN_LINEAR_FC2_WEIGHT,
  IN_LINEAR_FC2_BIAS,
  IN_VISION_Q_WEIGHT,
  IN_VISION_Q_BIAS,
  IN_VISION_K_WEIGHT,
  IN_VISION_K_BIAS,
  IN_VISION_V_WEIGHT,
  IN_VISION_V_BIAS
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_INPUT_NORM_BIAS, "norm1.bias"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_POST_NORM_BIAS, "norm2.bias"},
    {IN_QKV_WEIGHT, "attn.qkv.weight"},
    {IN_QKV_BIAS, "attn.qkv.bias"},
    {IN_WATTENTION_OUT_WEIGHT, "attn.proj.weight"},
    {IN_WATTENTION_OUT_BIAS, "attn.proj.bias"},
    {IN_LINEAR_FC1_WEIGHT, "mlp.fc1.weight"},
    {IN_LINEAR_FC1_BIAS, "mlp.fc1.bias"},
    {IN_LINEAR_FC2_WEIGHT, "mlp.fc2.weight"},
    {IN_LINEAR_FC2_BIAS, "mlp.fc2.bias"}};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {IN_WATTENTION_OUT_WEIGHT, 1},
    {IN_LINEAR_FC1_WEIGHT, 0},
    {IN_LINEAR_FC1_BIAS, 0},
    {IN_LINEAR_FC2_WEIGHT, 1},
};

Qwen2VisionEncoderLoader::Qwen2VisionEncoderLoader(uint64_t weight_count,
                                                   const ModelContext& context)
    : BaseLoader(weight_count, context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  encode_param_rank = parallel_args.rank();
  encode_param_worldSize = parallel_args.world_size();
  at_weight_tensors_.resize(weight_count);
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen2VisionEncoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

void Qwen2VisionEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen2VisionEncoderLoader::merge_loaded_weights() {
  // spilt pack qkv weight when enable tp
  get_weights_col_packed_qkv();
  if (encode_param_worldSize > 1) {
    // merge qkv weight
    auto new_qkv_weight = torch::cat({at_weight_tensors_[IN_VISION_Q_WEIGHT],
                                      at_weight_tensors_[IN_VISION_K_WEIGHT],
                                      at_weight_tensors_[IN_VISION_V_WEIGHT]},
                                     0);
    at_weight_tensors_[IN_QKV_WEIGHT] = new_qkv_weight;
    at_weight_tensors_[IN_VISION_Q_WEIGHT] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_K_WEIGHT] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_V_WEIGHT] = torch::zeros({1}).to(device_);

    // merge qkv bias
    auto new_qkv_bias = torch::cat({at_weight_tensors_[IN_VISION_Q_BIAS],
                                    at_weight_tensors_[IN_VISION_K_BIAS],
                                    at_weight_tensors_[IN_VISION_V_BIAS]},
                                   0);
    at_weight_tensors_[IN_QKV_BIAS] = new_qkv_bias;
    at_weight_tensors_[IN_VISION_Q_BIAS] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_K_BIAS] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_VISION_V_BIAS] = torch::zeros({1}).to(device_);
  }
}

// tp spilt weight
void Qwen2VisionEncoderLoader::get_weights_col_packed_qkv() {
  int rank = encode_param_rank;
  int worldSize = encode_param_worldSize;
  // split qkv weight
  qkv_weight = torch::chunk(at_weight_tensors_[IN_QKV_WEIGHT], 3, 0);
  qkv_bias = torch::chunk(at_weight_tensors_[IN_QKV_BIAS], 3, 0);
  // weight
  at_weight_tensors_[IN_VISION_Q_WEIGHT] =
      (qkv_weight[0].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_VISION_K_WEIGHT] =
      (qkv_weight[1].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_VISION_V_WEIGHT] =
      (qkv_weight[2].chunk(worldSize, 0))[rank];
  // bias
  at_weight_tensors_[IN_VISION_Q_BIAS] =
      (qkv_bias[0].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_VISION_K_BIAS] =
      (qkv_bias[1].chunk(worldSize, 0))[rank];
  at_weight_tensors_[IN_VISION_V_BIAS] =
      (qkv_bias[2].chunk(worldSize, 0))[rank];
}

}  // namespace layer
}  // namespace xllm