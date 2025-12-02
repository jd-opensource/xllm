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

#include "qwen2dot5_vision_encoder_loader.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUException.h"

namespace xllm {
namespace layer {

enum VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_POST_NORM_WEIGHT,
  IN_QKV_WEIGHT,
  IN_QKV_BIAS,
  IN_WATTENTION_OUT_WEIGHT,
  IN_WATTENTION_OUT_BIAS,
  IN_MLP_GATE_WEIGHT,
  IN_MLP_GATE_BIAS,
  IN_MLP_UP_WEIGHT,
  IN_MLP_UP_BIAS,
  IN_MLP_DOWN_WEIGHT,
  IN_MLP_DOWN_BIAS,
  IN_VISION_Q_WEIGHT,
  IN_VISION_Q_BIAS,
  IN_VISION_K_WEIGHT,
  IN_VISION_K_BIAS,
  IN_VISION_V_WEIGHT,
  IN_VISION_V_BIAS
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_QKV_WEIGHT, "qkv.weight"},
    {IN_QKV_BIAS, "qkv.bias"},
    {IN_WATTENTION_OUT_WEIGHT, "attn.proj.weight"},
    {IN_WATTENTION_OUT_BIAS, "attn.proj.bias"},
    {IN_MLP_GATE_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_GATE_BIAS, "mlp.gate_proj.bias"},
    {IN_MLP_UP_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_UP_BIAS, "mlp.up_proj.bias"},
    {IN_MLP_DOWN_WEIGHT, "mlp.down_proj.weight"},
    {IN_MLP_DOWN_BIAS, "mlp.down_proj.bias"},
};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {IN_WATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATE_WEIGHT, 0},
    {IN_MLP_GATE_BIAS, 0},
    {IN_MLP_UP_WEIGHT, 0},
    {IN_MLP_UP_BIAS, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

Qwen2dot5VisionEncoderLoader::Qwen2dot5VisionEncoderLoader(
    uint64_t weight_count,
    const ModelContext& context,
    int64_t numAttentionHeadsPerRank)
    : BaseLoader(weight_count, context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  encode_param_rank = parallel_args.rank();
  encode_param_worldSize = parallel_args.world_size();
  encode_param_numAttentionHeadsPerRank = numAttentionHeadsPerRank;
  at_weight_tensors_.resize(weight_count);
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  device_id_ = options.device().index();
  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen2dot5VisionEncoderLoader::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
  get_weights_col_packed_qkv();
}

// tp spilt weight
void Qwen2dot5VisionEncoderLoader::get_weights_col_packed_qkv() {
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

void Qwen2dot5VisionEncoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen2dot5VisionEncoderLoader::merge_loaded_weights() {
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
  // pad qkv weights
  pad_qkv_weights();
  // merge gate up
  auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_GATE_WEIGHT],
                                    at_weight_tensors_[IN_MLP_UP_WEIGHT]},
                                   0);
  at_weight_tensors_[IN_MLP_GATE_WEIGHT] = new_mlp_weight;
  auto new_mlp_bias = torch::cat({at_weight_tensors_[IN_MLP_GATE_BIAS],
                                  at_weight_tensors_[IN_MLP_UP_BIAS]},
                                 0);
  at_weight_tensors_[IN_MLP_GATE_BIAS] = new_mlp_bias;
  at_weight_tensors_[IN_MLP_UP_BIAS] = torch::zeros({1}).to(device_);
  // pad mlp weights
  pad_mlp_weights();
}

void Qwen2dot5VisionEncoderLoader::pad_qkv_weights() {
  auto qkv_proj_weight = at_weight_tensors_[IN_QKV_WEIGHT];
  auto qkv_proj_bias = at_weight_tensors_[IN_QKV_BIAS];
  int num_heads_pre_rank = encode_param_numAttentionHeadsPerRank;
  int hidden_size = num_heads_pre_rank * 80 * encode_param_worldSize;

  auto qkv_proj_weight_reshaped =
      qkv_proj_weight.reshape({num_heads_pre_rank, 3, 80, hidden_size});

  auto first_half = qkv_proj_weight_reshaped.slice(2, 0, 40);
  auto second_half = qkv_proj_weight_reshaped.slice(2, 40, 80);

  auto first_half_padded = torch::nn::functional::pad(
      first_half, torch::nn::functional::PadFuncOptions({0, 0, 0, 24}));
  auto second_half_padded = torch::nn::functional::pad(
      second_half, torch::nn::functional::PadFuncOptions({0, 0, 0, 24}));

  auto qkv_proj_weight_padded =
      torch::cat({first_half_padded, second_half_padded}, 2);
  auto qkv_proj_weight_final = qkv_proj_weight_padded.reshape(
      {num_heads_pre_rank * 128 * 3, hidden_size});
  qkv_proj_weight_final =
      at_npu::native::npu_format_cast(qkv_proj_weight_final, 2);

  auto qkv_proj_bias_reshaped =
      qkv_proj_bias.reshape({num_heads_pre_rank, 3, 80});
  first_half = qkv_proj_bias_reshaped.slice(2, 0, 40);
  second_half = qkv_proj_bias_reshaped.slice(2, 40, 80);

  first_half_padded = torch::nn::functional::pad(
      first_half, torch::nn::functional::PadFuncOptions({0, 24}));
  second_half_padded = torch::nn::functional::pad(
      second_half, torch::nn::functional::PadFuncOptions({0, 24}));
  auto qkv_proj_bias_padded =
      torch::cat({first_half_padded, second_half_padded}, 2);
  auto qkv_proj_bias_final =
      qkv_proj_bias_padded.reshape({num_heads_pre_rank * 128 * 3});

  at_weight_tensors_[IN_QKV_WEIGHT] = qkv_proj_weight_final;
  at_weight_tensors_[IN_QKV_BIAS] = qkv_proj_bias_final;

  auto out_proj_weight = at_weight_tensors_[IN_WATTENTION_OUT_WEIGHT];

  out_proj_weight =
      torch::nn::functional::pad(
          out_proj_weight.reshape({hidden_size, num_heads_pre_rank * 2, 40}),
          torch::nn::functional::PadFuncOptions({0, 24, 0, 0}))
          .reshape({hidden_size, num_heads_pre_rank * 128});
  at_weight_tensors_[IN_WATTENTION_OUT_WEIGHT] = out_proj_weight;
}

void Qwen2dot5VisionEncoderLoader::pad_mlp_weights() {
  torch::Tensor weight = at_weight_tensors_[IN_MLP_GATE_WEIGHT];
  torch::Tensor bias = at_weight_tensors_[IN_MLP_GATE_BIAS];

  int64_t tp_intermediate_size_half = weight.size(0) / 2;
  int64_t remainder = tp_intermediate_size_half % 32;
  int64_t tp_intermediate_size_half_pad;
  if (remainder != 0) {
    tp_intermediate_size_half_pad =
        tp_intermediate_size_half + (32 - remainder);
  } else {
    tp_intermediate_size_half_pad = tp_intermediate_size_half;
  }
  auto weight_split1 = weight.slice(0, 0, tp_intermediate_size_half);
  auto weight_split2 = weight.slice(0, tp_intermediate_size_half);
  auto bias_split1 = bias.slice(0, 0, tp_intermediate_size_half);
  auto bias_split2 = bias.slice(0, tp_intermediate_size_half);

  auto weight_split1_padded =
      pad_tensor(weight_split1, tp_intermediate_size_half_pad);
  auto weight_split2_padded =
      pad_tensor(weight_split2, tp_intermediate_size_half_pad);
  auto bias_split1_padded =
      pad_tensor(bias_split1, tp_intermediate_size_half_pad);
  auto bias_split2_padded =
      pad_tensor(bias_split2, tp_intermediate_size_half_pad);

  auto weight_padded =
      torch::cat({weight_split1_padded, weight_split2_padded}, 0);
  auto bias_padded = torch::cat({bias_split1_padded, bias_split2_padded}, 0);
  at_weight_tensors_[IN_MLP_GATE_WEIGHT] = weight_padded;
  at_weight_tensors_[IN_MLP_GATE_BIAS] = bias_padded;

  torch::Tensor down_weight = at_weight_tensors_[IN_MLP_DOWN_WEIGHT];

  auto tp_intermediate_size = down_weight.size(1);
  remainder = tp_intermediate_size % 32;
  int64_t tp_intermediate_size_pad;
  if (remainder != 0) {
    tp_intermediate_size_pad = tp_intermediate_size + (32 - remainder);
  } else {
    tp_intermediate_size_pad = tp_intermediate_size;
  }

  auto down_weight_padded =
      pad_tensor(down_weight, tp_intermediate_size_pad, 1);
  at_weight_tensors_[IN_MLP_DOWN_WEIGHT] = down_weight_padded;
}

torch::Tensor Qwen2dot5VisionEncoderLoader::pad_tensor(
    const torch::Tensor& tensor,
    int64_t target_shape,
    int64_t dim) {
  int64_t pad_size = target_shape - tensor.size(dim);
  if (tensor.dim() == 1) {
    return torch::nn::functional::pad(
        tensor, torch::nn::functional::PadFuncOptions({0, pad_size}));
  } else if (tensor.dim() == 2) {
    if (1 == dim)
      return torch::nn::functional::pad(
          tensor, torch::nn::functional::PadFuncOptions({0, pad_size, 0, 0}));
    else
      return torch::nn::functional::pad(
          tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  }
  return tensor;
}

}  // namespace layer
}  // namespace xllm