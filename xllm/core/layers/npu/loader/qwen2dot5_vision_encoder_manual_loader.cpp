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

#include "qwen2dot5_vision_encoder_manual_loader.h"

#include "qwen_loader_constants.h"

namespace xllm {
namespace layer {

using namespace qwen2dot5_vision_encoder_constants;

Qwen2dot5VisionEncoderManualLoader::Qwen2dot5VisionEncoderManualLoader(
    uint64_t weight_count,
    const ModelContext& context,
    int64_t numAttentionHeadsPerRank)
    : BaseManualLoader(weight_count, context) {
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();
  auto host_options =
      torch::TensorOptions().dtype(options.dtype()).device(torch::kCPU);

  encode_param_rank_ = parallel_args.rank();
  encode_param_world_size_ = parallel_args.world_size();
  encode_param_num_attention_heads_per_rank_ = numAttentionHeadsPerRank;
  at_host_weight_tensors_.resize(weight_count);
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  for (int i = 0; i < weight_count; ++i) {
    at_host_weight_tensors_[i] = torch::zeros({1}, host_options);
  }
}

void Qwen2dot5VisionEncoderManualLoader::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], true);
    } else {
      set_weight(state_dict, name, index, true);
    }
  }
}

void Qwen2dot5VisionEncoderManualLoader::get_weights_col_packed_qkv() {
  qkv_weight_ = torch::chunk(at_host_weight_tensors_[IN_QKV_WEIGHT], 3, 0);
  qkv_bias_ = torch::chunk(at_host_weight_tensors_[IN_QKV_BIAS], 3, 0);

  at_host_weight_tensors_[IN_VISION_Q_WEIGHT] =
      (qkv_weight_[0].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  at_host_weight_tensors_[IN_VISION_K_WEIGHT] =
      (qkv_weight_[1].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  at_host_weight_tensors_[IN_VISION_V_WEIGHT] =
      (qkv_weight_[2].chunk(encode_param_world_size_, 0))[encode_param_rank_];

  at_host_weight_tensors_[IN_VISION_Q_BIAS] =
      (qkv_bias_[0].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  at_host_weight_tensors_[IN_VISION_K_BIAS] =
      (qkv_bias_[1].chunk(encode_param_world_size_, 0))[encode_param_rank_];
  at_host_weight_tensors_[IN_VISION_V_BIAS] =
      (qkv_bias_[2].chunk(encode_param_world_size_, 0))[encode_param_rank_];
}

void Qwen2dot5VisionEncoderManualLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_host_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen2dot5VisionEncoderManualLoader::merge_host_at_weights() {
  auto make_zero_like = [](const torch::Tensor& ref) {
    return torch::zeros(
        {1},
        torch::TensorOptions().dtype(ref.scalar_type()).device(torch::kCPU));
  };

  get_weights_col_packed_qkv();
  if (encode_param_world_size_ > 1) {
    at_host_weight_tensors_[IN_QKV_WEIGHT] =
        torch::cat({at_host_weight_tensors_[IN_VISION_Q_WEIGHT],
                    at_host_weight_tensors_[IN_VISION_K_WEIGHT],
                    at_host_weight_tensors_[IN_VISION_V_WEIGHT]},
                   0);
    at_host_weight_tensors_[IN_VISION_Q_WEIGHT] =
        make_zero_like(at_host_weight_tensors_[IN_VISION_Q_WEIGHT]);
    at_host_weight_tensors_[IN_VISION_K_WEIGHT] =
        make_zero_like(at_host_weight_tensors_[IN_VISION_K_WEIGHT]);
    at_host_weight_tensors_[IN_VISION_V_WEIGHT] =
        make_zero_like(at_host_weight_tensors_[IN_VISION_V_WEIGHT]);

    at_host_weight_tensors_[IN_QKV_BIAS] =
        torch::cat({at_host_weight_tensors_[IN_VISION_Q_BIAS],
                    at_host_weight_tensors_[IN_VISION_K_BIAS],
                    at_host_weight_tensors_[IN_VISION_V_BIAS]},
                   0);
    at_host_weight_tensors_[IN_VISION_Q_BIAS] =
        make_zero_like(at_host_weight_tensors_[IN_VISION_Q_BIAS]);
    at_host_weight_tensors_[IN_VISION_K_BIAS] =
        make_zero_like(at_host_weight_tensors_[IN_VISION_K_BIAS]);
    at_host_weight_tensors_[IN_VISION_V_BIAS] =
        make_zero_like(at_host_weight_tensors_[IN_VISION_V_BIAS]);
  }

  pad_qkv_weights();

  at_host_weight_tensors_[IN_MLP_GATE_WEIGHT] =
      torch::cat({at_host_weight_tensors_[IN_MLP_GATE_WEIGHT],
                  at_host_weight_tensors_[IN_MLP_UP_WEIGHT]},
                 0);
  at_host_weight_tensors_[IN_MLP_GATE_BIAS] =
      torch::cat({at_host_weight_tensors_[IN_MLP_GATE_BIAS],
                  at_host_weight_tensors_[IN_MLP_UP_BIAS]},
                 0);
  at_host_weight_tensors_[IN_MLP_UP_BIAS] =
      make_zero_like(at_host_weight_tensors_[IN_MLP_UP_BIAS]);

  pad_mlp_weights();
}

void Qwen2dot5VisionEncoderManualLoader::pad_qkv_weights() {
  auto qkv_proj_weight = at_host_weight_tensors_[IN_QKV_WEIGHT];
  auto qkv_proj_bias = at_host_weight_tensors_[IN_QKV_BIAS];
  int num_heads_per_rank = encode_param_num_attention_heads_per_rank_;
  int hidden_size = num_heads_per_rank * 80 * encode_param_world_size_;

  auto qkv_proj_weight_reshaped =
      qkv_proj_weight.reshape({num_heads_per_rank, 3, 80, hidden_size});

  auto first_half = qkv_proj_weight_reshaped.slice(2, 0, 40);
  auto second_half = qkv_proj_weight_reshaped.slice(2, 40, 80);

  auto first_half_padded = torch::nn::functional::pad(
      first_half, torch::nn::functional::PadFuncOptions({0, 0, 0, 24}));
  auto second_half_padded = torch::nn::functional::pad(
      second_half, torch::nn::functional::PadFuncOptions({0, 0, 0, 24}));

  auto qkv_proj_weight_padded =
      torch::cat({first_half_padded, second_half_padded}, 2);
  auto qkv_proj_weight_final = qkv_proj_weight_padded.reshape(
      {num_heads_per_rank * 128 * 3, hidden_size});

  auto qkv_proj_bias_reshaped =
      qkv_proj_bias.reshape({num_heads_per_rank, 3, 80});
  first_half = qkv_proj_bias_reshaped.slice(2, 0, 40);
  second_half = qkv_proj_bias_reshaped.slice(2, 40, 80);

  first_half_padded = torch::nn::functional::pad(
      first_half, torch::nn::functional::PadFuncOptions({0, 24}));
  second_half_padded = torch::nn::functional::pad(
      second_half, torch::nn::functional::PadFuncOptions({0, 24}));
  auto qkv_proj_bias_padded =
      torch::cat({first_half_padded, second_half_padded}, 2);
  auto qkv_proj_bias_final =
      qkv_proj_bias_padded.reshape({num_heads_per_rank * 128 * 3});

  at_host_weight_tensors_[IN_QKV_WEIGHT] = qkv_proj_weight_final;
  at_host_weight_tensors_[IN_QKV_BIAS] = qkv_proj_bias_final;

  auto out_proj_weight = at_host_weight_tensors_[IN_WATTENTION_OUT_WEIGHT];
  out_proj_weight =
      torch::nn::functional::pad(
          out_proj_weight.reshape({hidden_size, num_heads_per_rank * 2, 40}),
          torch::nn::functional::PadFuncOptions({0, 24, 0, 0}))
          .reshape({hidden_size, num_heads_per_rank * 128});
  at_host_weight_tensors_[IN_WATTENTION_OUT_WEIGHT] = out_proj_weight;
}

void Qwen2dot5VisionEncoderManualLoader::pad_mlp_weights() {
  auto weight = at_host_weight_tensors_[IN_MLP_GATE_WEIGHT];
  auto bias = at_host_weight_tensors_[IN_MLP_GATE_BIAS];

  int64_t tp_intermediate_size_half = weight.size(0) / 2;
  int64_t remainder = tp_intermediate_size_half % 32;
  int64_t tp_intermediate_size_half_pad =
      (remainder != 0) ? (tp_intermediate_size_half + (32 - remainder))
                       : tp_intermediate_size_half;

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

  at_host_weight_tensors_[IN_MLP_GATE_WEIGHT] =
      torch::cat({weight_split1_padded, weight_split2_padded}, 0);
  at_host_weight_tensors_[IN_MLP_GATE_BIAS] =
      torch::cat({bias_split1_padded, bias_split2_padded}, 0);

  auto down_weight = at_host_weight_tensors_[IN_MLP_DOWN_WEIGHT];
  auto tp_intermediate_size = down_weight.size(1);
  remainder = tp_intermediate_size % 32;
  int64_t tp_intermediate_size_pad =
      (remainder != 0) ? (tp_intermediate_size + (32 - remainder))
                       : tp_intermediate_size;

  at_host_weight_tensors_[IN_MLP_DOWN_WEIGHT] =
      pad_tensor(down_weight, tp_intermediate_size_pad, 1);
}

torch::Tensor Qwen2dot5VisionEncoderManualLoader::pad_tensor(
    const torch::Tensor& tensor,
    int64_t target_shape,
    int64_t dim) {
  int64_t pad_size = target_shape - tensor.size(dim);
  if (tensor.dim() == 1) {
    return torch::nn::functional::pad(
        tensor, torch::nn::functional::PadFuncOptions({0, pad_size}));
  }
  if (tensor.dim() == 2) {
    if (dim == 1) {
      return torch::nn::functional::pad(
          tensor, torch::nn::functional::PadFuncOptions({0, pad_size, 0, 0}));
    }
    return torch::nn::functional::pad(
        tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size}));
  }
  return tensor;
}

}  // namespace layer
}  // namespace xllm
