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

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <torch_npu/csrc/libs/init_npu.h>

#include "qwen3_audio_encoder_loader.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

enum AudioEncoderLayerTensorId : int {
  AUDIO_IN_INPUT_NORM_WEIGHT = 0,
  AUDIO_IN_INPUT_NORM_BIAS,
  AUDIO_IN_POST_NORM_WEIGHT,
  AUDIO_IN_POST_NORM_BIAS,
  AUDIO_IN_QKV_WEIGHT,
  AUDIO_IN_QKV_BIAS,
  AUDIO_IN_WATTENTION_OUT_WEIGHT,
  AUDIO_IN_WATTENTION_OUT_BIAS,
  AUDIO_IN_LINEAR_FC1_WEIGHT,
  AUDIO_IN_LINEAR_FC1_BIAS,
  AUDIO_IN_LINEAR_FC2_WEIGHT,
  AUDIO_IN_LINEAR_FC2_BIAS,
  AUDIO_IN_Q_WEIGHT,
  AUDIO_IN_Q_BIAS,
  AUDIO_IN_K_WEIGHT,
  AUDIO_IN_K_BIAS,
  AUDIO_IN_V_WEIGHT,
  AUDIO_IN_V_BIAS
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {AUDIO_IN_INPUT_NORM_WEIGHT, "self_attn_layer_norm.weight"},
    {AUDIO_IN_INPUT_NORM_BIAS, "self_attn_layer_norm.bias"},
    {AUDIO_IN_POST_NORM_WEIGHT, "final_layer_norm.weight"},
    {AUDIO_IN_POST_NORM_BIAS, "final_layer_norm.bias"},
    {AUDIO_IN_WATTENTION_OUT_WEIGHT, "self_attn.out_proj.weight"},
    {AUDIO_IN_WATTENTION_OUT_BIAS, "self_attn.out_proj.bias"},
    {AUDIO_IN_LINEAR_FC1_WEIGHT, "fc1.weight"},
    {AUDIO_IN_LINEAR_FC1_BIAS, "fc1.bias"},
    {AUDIO_IN_LINEAR_FC2_WEIGHT, "fc2.weight"},
    {AUDIO_IN_LINEAR_FC2_BIAS, "fc2.bias"},
    {AUDIO_IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {AUDIO_IN_Q_BIAS, "self_attn.q_proj.bias"},
    {AUDIO_IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {AUDIO_IN_K_BIAS, "self_attn.k_proj.bias"},
    {AUDIO_IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {AUDIO_IN_V_BIAS, "self_attn.v_proj.bias"}};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {AUDIO_IN_WATTENTION_OUT_WEIGHT, 1},
    {AUDIO_IN_LINEAR_FC1_WEIGHT, 0},
    {AUDIO_IN_LINEAR_FC1_BIAS, 0},
    {AUDIO_IN_LINEAR_FC2_WEIGHT, 1},
};

Qwen3AudioEncoderLoader::Qwen3AudioEncoderLoader(uint64_t weight_count,
                                                 const ModelContext& context)
    : BaseLoader(weight_count, context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  encode_param_rank_ = parallel_args.rank();
  encode_param_worldSize_ = parallel_args.world_size();
  at_weight_tensors_.resize(weight_count);
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen3AudioEncoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

void Qwen3AudioEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3AudioEncoderLoader::merge_loaded_weights() {
  // spilt pack qkv weight when enable tp
  get_weights_col_packed_qkv();

  // merge qkv weight
  auto new_qkv_weight = torch::cat({at_weight_tensors_[AUDIO_IN_Q_WEIGHT],
                                    at_weight_tensors_[AUDIO_IN_K_WEIGHT],
                                    at_weight_tensors_[AUDIO_IN_V_WEIGHT]},
                                   0)
                            .to(device_);
  at_weight_tensors_[AUDIO_IN_QKV_WEIGHT] = new_qkv_weight;
  at_weight_tensors_[AUDIO_IN_Q_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[AUDIO_IN_K_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[AUDIO_IN_V_WEIGHT] = torch::zeros({1}).to(device_);

  // merge qkv bias
  auto new_qkv_bias = torch::cat({at_weight_tensors_[AUDIO_IN_Q_BIAS],
                                  at_weight_tensors_[AUDIO_IN_K_BIAS],
                                  at_weight_tensors_[AUDIO_IN_V_BIAS]},
                                 0)
                          .to(device_);

  at_weight_tensors_[AUDIO_IN_QKV_BIAS] = new_qkv_bias;
  at_weight_tensors_[AUDIO_IN_Q_BIAS] = torch::zeros({1}).to(device_);
  at_weight_tensors_[AUDIO_IN_K_BIAS] = torch::zeros({1}).to(device_);
  at_weight_tensors_[AUDIO_IN_V_BIAS] = torch::zeros({1}).to(device_);
}

// tp spilt weight
void Qwen3AudioEncoderLoader::get_weights_col_packed_qkv() {
  int rank = encode_param_rank_;
  int worldSize = encode_param_worldSize_;
  // weight
  at_weight_tensors_[AUDIO_IN_Q_WEIGHT] =
      (at_weight_tensors_[AUDIO_IN_Q_WEIGHT].chunk(worldSize, 0))[rank].to(
          device_);
  at_weight_tensors_[AUDIO_IN_K_WEIGHT] =
      (at_weight_tensors_[AUDIO_IN_K_WEIGHT].chunk(worldSize, 0))[rank].to(
          device_);
  at_weight_tensors_[AUDIO_IN_V_WEIGHT] =
      (at_weight_tensors_[AUDIO_IN_V_WEIGHT].chunk(worldSize, 0))[rank].to(
          device_);
  // bias
  at_weight_tensors_[AUDIO_IN_Q_BIAS] =
      (at_weight_tensors_[AUDIO_IN_Q_BIAS].chunk(worldSize, 0))[rank].to(
          device_);
  at_weight_tensors_[AUDIO_IN_K_BIAS] =
      (at_weight_tensors_[AUDIO_IN_K_BIAS].chunk(worldSize, 0))[rank].to(
          device_);
  at_weight_tensors_[AUDIO_IN_V_BIAS] =
      (at_weight_tensors_[AUDIO_IN_V_BIAS].chunk(worldSize, 0))[rank].to(
          device_);
}

}  // namespace layer
}  // namespace xllm
