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

#include "qwen2_decoder_manual_loader.h"

namespace xllm {
namespace layer {

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_Q_BIAS, "self_attn.q_proj.bias"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_K_BIAS, "self_attn.k_proj.bias"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_V_BIAS, "self_attn.v_proj.bias"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"}};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING_W8A8 = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_Q_BIAS, "self_attn.q_proj.quant_bias"},
    {IN_Q_DEQSCALE, "self_attn.q_proj.deq_scale"},
    {IN_Q_OFFSET, "self_attn.q_proj.input_offset"},
    {IN_Q_SCALE, "self_attn.q_proj.input_scale"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_K_BIAS, "self_attn.k_proj.quant_bias"},
    {IN_K_DEQSCALE, "self_attn.k_proj.deq_scale"},
    {IN_K_OFFSET, "self_attn.k_proj.input_offset"},
    {IN_K_SCALE, "self_attn.k_proj.input_scale"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_V_BIAS, "self_attn.v_proj.quant_bias"},
    {IN_V_DEQSCALE, "self_attn.v_proj.deq_scale"},
    {IN_V_OFFSET, "self_attn.v_proj.input_offset"},
    {IN_V_SCALE, "self_attn.v_proj.input_scale"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_ATTENTION_OUT_BIAS, "self_attn.o_proj.quant_bias"},
    {IN_ATTENTION_OUT_DEQSCALE, "self_attn.o_proj.deq_scale"},
    {IN_ATTENTION_OUT_OFFSET, "self_attn.o_proj.input_offset"},
    {IN_ATTENTION_OUT_SCALE, "self_attn.o_proj.input_scale"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W2_BIAS, "mlp.gate_proj.quant_bias"},
    {IN_MLP_W2_DEQSCALE, "mlp.gate_proj.deq_scale"},
    {IN_MLP_W2_OFFSET, "mlp.gate_proj.input_offset"},
    {IN_MLP_W2_SCALE, "mlp.gate_proj.input_scale"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_W1_BIAS, "mlp.up_proj.quant_bias"},
    {IN_MLP_W1_DEQSCALE, "mlp.up_proj.deq_scale"},
    {IN_MLP_W1_OFFSET, "mlp.up_proj.input_offset"},
    {IN_MLP_W1_SCALE, "mlp.up_proj.input_scale"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"}};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_Q_BIAS, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_K_BIAS, 0},
                                          {IN_V_WEIGHT, 0},
                                          {IN_V_BIAS, 0},
                                          {IN_ATTENTION_OUT_WEIGHT, 1},
                                          {IN_MLP_W2_WEIGHT, 0},
                                          {IN_MLP_W1_WEIGHT, 0},
                                          {IN_MLP_CPROJ_WEIGHT, 1}};

static std::map<int, int> WEIGHT_SHARD_W8A8 = {{IN_Q_WEIGHT, 0},
                                               {IN_Q_BIAS, 0},
                                               {IN_Q_DEQSCALE, 0},
                                               {IN_K_WEIGHT, 0},
                                               {IN_K_BIAS, 0},
                                               {IN_K_DEQSCALE, 0},
                                               {IN_V_WEIGHT, 0},
                                               {IN_V_BIAS, 0},
                                               {IN_V_DEQSCALE, 0},
                                               {IN_ATTENTION_OUT_WEIGHT, 1},
                                               {IN_MLP_W2_WEIGHT, 0},
                                               {IN_MLP_W2_BIAS, 0},
                                               {IN_MLP_W2_DEQSCALE, 0},
                                               {IN_MLP_W1_WEIGHT, 0},
                                               {IN_MLP_W1_BIAS, 0},
                                               {IN_MLP_W1_DEQSCALE, 0},
                                               {IN_MLP_CPROJ_WEIGHT, 1}};

Qwen2DecoderManualLoader::Qwen2DecoderManualLoader(uint64_t weight_count,
                                                   const ModelContext& context)
    : BaseManualLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  device_id_ = options.device().index();

  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen2DecoderManualLoader::load_state_dict(const StateDict& state_dict) {
  if (quantize_type_ == "w8a8") {
    for (const auto& [index, name] : WEIGHT_MAPPING_W8A8) {
      if (WEIGHT_SHARD_W8A8.find(index) != WEIGHT_SHARD_W8A8.end()) {
        set_weight(state_dict, name, index, WEIGHT_SHARD_W8A8[index], true);
      } else {
        set_weight(state_dict, name, index, true);
      }
    }
    at_host_weight_tensors_[IN_NORM_BIAS] =
        torch::zeros(at_host_weight_tensors_[IN_NORM_WEIGHT].sizes(),
                     at_host_weight_tensors_[IN_NORM_WEIGHT].options());

    at_host_weight_tensors_[IN_SELFOUT_NORM_BIAS] =
        torch::zeros(at_host_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].sizes(),
                     at_host_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].options());

    return;
  }

  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], true);
    } else {
      set_weight(state_dict, name, index, true);
    }
  }
}

void Qwen2DecoderManualLoader::merge_loaded_weights() {
  merge_host_at_weights();
  init_weight_slices();
  copy_weights_to_device();
  init_device_at_weights();
}

void Qwen2DecoderManualLoader::merge_and_move_pinned_host() {
  merge_host_at_weights();
  init_weight_slices();
  copy_weights_to_pinned_host();
}

void Qwen2DecoderManualLoader::merge_host_at_weights() {
  auto make_zero_like = [](const torch::Tensor& ref) {
    return torch::zeros(
        {1},
        torch::TensorOptions().dtype(ref.scalar_type()).device(torch::kCPU));
  };

  if (quantize_type_ == "w8a8") {
    at_host_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE] =
        at_host_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
    at_host_weight_tensors_[IN_Q_DEQSCALE] =
        torch::cat({at_host_weight_tensors_[IN_Q_DEQSCALE],
                    at_host_weight_tensors_[IN_K_DEQSCALE],
                    at_host_weight_tensors_[IN_V_DEQSCALE]},
                   0)
            .to(torch::kFloat32);
    at_host_weight_tensors_[IN_K_DEQSCALE] =
        make_zero_like(at_host_weight_tensors_[IN_K_DEQSCALE]);
    at_host_weight_tensors_[IN_V_DEQSCALE] =
        make_zero_like(at_host_weight_tensors_[IN_V_DEQSCALE]);
    at_host_weight_tensors_[IN_K_OFFSET] =
        make_zero_like(at_host_weight_tensors_[IN_K_OFFSET]);
    at_host_weight_tensors_[IN_V_OFFSET] =
        make_zero_like(at_host_weight_tensors_[IN_V_OFFSET]);
    at_host_weight_tensors_[IN_K_SCALE] =
        make_zero_like(at_host_weight_tensors_[IN_K_SCALE]);
    at_host_weight_tensors_[IN_V_SCALE] =
        make_zero_like(at_host_weight_tensors_[IN_V_SCALE]);
    at_host_weight_tensors_[IN_MLP_W2_BIAS] =
        torch::cat({at_host_weight_tensors_[IN_MLP_W2_BIAS],
                    at_host_weight_tensors_[IN_MLP_W1_BIAS]},
                   0);
    at_host_weight_tensors_[IN_MLP_W1_BIAS] =
        make_zero_like(at_host_weight_tensors_[IN_MLP_W1_BIAS]);
    at_host_weight_tensors_[IN_MLP_W2_DEQSCALE] =
        torch::cat({at_host_weight_tensors_[IN_MLP_W2_DEQSCALE],
                    at_host_weight_tensors_[IN_MLP_W1_DEQSCALE]},
                   0)
            .to(torch::kFloat32);
    at_host_weight_tensors_[IN_MLP_W1_DEQSCALE] =
        make_zero_like(at_host_weight_tensors_[IN_MLP_W1_DEQSCALE]);
    at_host_weight_tensors_[IN_MLP_W1_OFFSET] =
        make_zero_like(at_host_weight_tensors_[IN_MLP_W1_OFFSET]);
    at_host_weight_tensors_[IN_MLP_W1_SCALE] =
        make_zero_like(at_host_weight_tensors_[IN_MLP_W1_SCALE]);
    at_host_weight_tensors_[IN_Q_OFFSET] =
        at_host_weight_tensors_[IN_Q_OFFSET].to(torch::kInt8);
    at_host_weight_tensors_[IN_ATTENTION_OUT_OFFSET] =
        at_host_weight_tensors_[IN_ATTENTION_OUT_OFFSET].to(torch::kInt8);
    at_host_weight_tensors_[IN_MLP_W2_OFFSET] =
        at_host_weight_tensors_[IN_MLP_W2_OFFSET].to(torch::kInt8);
    if (device_id_ != 0) {
      torch::Tensor original_tensor =
          at_host_weight_tensors_[IN_ATTENTION_OUT_BIAS];
      auto shape = original_tensor.sizes();
      auto dtype = original_tensor.dtype();

      at_host_weight_tensors_[IN_ATTENTION_OUT_BIAS] = torch::zeros(
          shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    }
  }

  auto new_q_weight = torch::cat({at_host_weight_tensors_[IN_Q_WEIGHT],
                                  at_host_weight_tensors_[IN_K_WEIGHT],
                                  at_host_weight_tensors_[IN_V_WEIGHT]},
                                 0);

  at_host_weight_tensors_[IN_Q_WEIGHT] = new_q_weight;

  at_host_weight_tensors_[IN_K_WEIGHT] =
      make_zero_like(at_host_weight_tensors_[IN_K_WEIGHT]);
  at_host_weight_tensors_[IN_V_WEIGHT] =
      make_zero_like(at_host_weight_tensors_[IN_V_WEIGHT]);

  auto new_q_bias = torch::cat({at_host_weight_tensors_[IN_Q_BIAS],
                                at_host_weight_tensors_[IN_K_BIAS],
                                at_host_weight_tensors_[IN_V_BIAS]},
                               0);
  at_host_weight_tensors_[IN_Q_BIAS] = new_q_bias;

  at_host_weight_tensors_[IN_K_BIAS] =
      make_zero_like(at_host_weight_tensors_[IN_K_BIAS]);
  at_host_weight_tensors_[IN_V_BIAS] =
      make_zero_like(at_host_weight_tensors_[IN_V_BIAS]);

  TransposeType transpose_type =
      check_transpose(at_host_weight_tensors_[IN_MLP_W2_WEIGHT]);
  if (transpose_type == TransposeType::TRANSPOSE) {
    auto new_mlp_weight =
        torch::cat({at_host_weight_tensors_[IN_MLP_W2_WEIGHT],
                    at_host_weight_tensors_[IN_MLP_W1_WEIGHT]},
                   0);
    at_host_weight_tensors_[IN_MLP_W2_WEIGHT] = new_mlp_weight.contiguous();
  } else {
    auto new_mlp_weight =
        torch::cat({at_host_weight_tensors_[IN_MLP_W2_WEIGHT],
                    at_host_weight_tensors_[IN_MLP_W1_WEIGHT]},
                   0)
            .transpose(0, 1);
    at_host_weight_tensors_[IN_MLP_W2_WEIGHT] = new_mlp_weight.contiguous();
  }

  at_host_weight_tensors_[IN_MLP_W1_WEIGHT] =
      make_zero_like(at_host_weight_tensors_[IN_MLP_W1_WEIGHT]);
}

TransposeType Qwen2DecoderManualLoader::check_transpose(at::Tensor& tensor) {
  bool is_k_divisible = tensor.size(1) % 256 == 0;
  bool is_n_divisible = tensor.size(0) % 256 == 0;

  if (!is_k_divisible && is_n_divisible) {
    return TransposeType::NOT_TRANSPOSE;
  }

  return TransposeType::TRANSPOSE;
}

void Qwen2DecoderManualLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_host_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

}  // namespace layer
}  // namespace xllm