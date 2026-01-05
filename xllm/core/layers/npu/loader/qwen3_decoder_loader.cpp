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

#include "qwen3_decoder_loader.h"

namespace xllm {
namespace layer {

enum DecoderLayerTensorId : int {
  IN_NORM_WEIGHT = 0,      // weight
  IN_NORM_BIAS = 1,        // bias
  IN_NORM_NEW_WEIGHT = 2,  // new weight
  IN_NORM_NEW_BIAS = 3,    // new bias

  IN_Q_WEIGHT = 4,    // weight
  IN_Q_BIAS = 5,      // bias
  IN_Q_DEQSCALE = 6,  // deq_scale
  IN_Q_OFFSET = 7,    // offset
  IN_Q_SCALE = 8,     // scale
  IN_Q_COMPRESS_IDX = 9,

  IN_K_WEIGHT = 10,    // weight
  IN_K_BIAS = 11,      // bias
  IN_K_DEQSCALE = 12,  // deq_scale
  IN_K_OFFSET = 13,    // offset
  IN_K_SCALE = 14,     // scale
  IN_K_COMPRESS_IDX = 15,

  IN_V_WEIGHT = 16,    // weight
  IN_V_BIAS = 17,      // bias
  IN_V_DEQSCALE = 18,  // deq_scale
  IN_V_OFFSET = 19,    // offset
  IN_V_SCALE = 20,     // scale
  IN_V_COMPRESS_IDX = 21,

  IN_ATTENTION_OUT_WEIGHT = 22,    // weight
  IN_ATTENTION_OUT_BIAS = 23,      // bias
  IN_ATTENTION_OUT_DEQSCALE = 24,  // deq_scale
  IN_ATTENTION_OUT_OFFSET = 25,    // offset
  IN_ATTENTION_OUT_SCALE = 26,     // scale
  IN_ATTENTION_OUT_COMPRESS_IDX = 27,

  IN_SELFOUT_NORM_WEIGHT = 28,      // weight
  IN_SELFOUT_NORM_BIAS = 29,        // bias
  IN_SELFOUT_NORM_NEW_WEIGHT = 30,  // new weight
  IN_SELFOUT_NORM_NEW_BIAS = 31,    // new bias

  IN_MLP_W2_WEIGHT = 32,    // weight
  IN_MLP_W2_BIAS = 33,      // bias
  IN_MLP_W2_DEQSCALE = 34,  // deq_scale
  IN_MLP_W2_OFFSET = 35,    // offset
  IN_MLP_W2_SCALE = 36,     // scale
  IN_MLP_W2_COMPRESS_IDX = 37,

  IN_MLP_W1_WEIGHT = 38,    // weight
  IN_MLP_W1_BIAS = 39,      // bias
  IN_MLP_W1_DEQSCALE = 40,  // deq_scale
  IN_MLP_W1_OFFSET = 41,    // offset
  IN_MLP_W1_SCALE = 42,     // scale
  IN_MLP_W1_COMPRESS_IDX = 43,

  IN_MLP_CPROJ_WEIGHT = 44,    // weight
  IN_MLP_CPROJ_BIAS = 45,      // bias
  IN_MLP_CPROJ_DEQSCALE = 46,  // deq_scale
  IN_MLP_CPROJ_OFFSET = 47,    // offset
  IN_MLP_CPROJ_SCALE = 48,     // scale
  IN_MLP_CPROJ_COMPRESS_IDX = 49,

  IN_QKV_SCALE_FILL = 50,
  IN_QKV_OFFSET_FILL = 51,
  IN_MLP_SCALE_FILL = 52,
  IN_MLP_OFFSET_FILL = 53,
  Q_NORM_WEIGHT = 54,
  K_NORM_WEIGHT = 55,
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {Q_NORM_WEIGHT, "self_attn.q_norm.weight"},
    {K_NORM_WEIGHT, "self_attn.k_norm.weight"}};

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
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {Q_NORM_WEIGHT, "self_attn.q_norm.weight"},
    {K_NORM_WEIGHT, "self_attn.k_norm.weight"}};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_V_WEIGHT, 0},
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

Qwen3DecoderLoader::Qwen3DecoderLoader(uint64_t weight_count,
                                       const ModelContext& context,
                                       bool enableAddNorm)
    : BaseLoader(weight_count, context), enableAddNorm_(enableAddNorm) {
  auto options = context.get_tensor_options();
  rank_id_ = parallel_args_.rank();

  dtype_ = torch::typeMetaToScalarType(options.dtype());
  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }

  at_placeholder_ = torch::zeros({1}).to(device_).to(dtype_);
}

void Qwen3DecoderLoader::load_state_dict(const StateDict& state_dict) {
  if (quantize_type_.compare("w8a8") == 0) {
    for (const auto& [index, name] : WEIGHT_MAPPING_W8A8) {
      if (WEIGHT_SHARD_W8A8.find(index) != WEIGHT_SHARD_W8A8.end()) {
        set_weight(state_dict, name, index, WEIGHT_SHARD_W8A8[index]);
      } else {
        set_weight(state_dict, name, index);
      }
    }
    at_weight_tensors_[IN_NORM_BIAS] =
        torch::zeros(at_weight_tensors_[IN_NORM_WEIGHT].sizes(),
                     at_weight_tensors_[IN_NORM_WEIGHT].options())
            .to(device_);

    at_weight_tensors_[IN_SELFOUT_NORM_BIAS] =
        torch::zeros(at_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].sizes(),
                     at_weight_tensors_[IN_SELFOUT_NORM_WEIGHT].options())
            .to(device_);
    return;
  }

  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

void Qwen3DecoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3DecoderLoader::merge_loaded_weights() {
  if (quantize_type_.compare("w8a8") == 0) {
    at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE] =
        at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
    at_weight_tensors_[IN_Q_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_Q_DEQSCALE],
                    at_weight_tensors_[IN_K_DEQSCALE],
                    at_weight_tensors_[IN_V_DEQSCALE]},
                   0)
            .to(torch::kFloat32);

    at_weight_tensors_[IN_Q_BIAS] = torch::cat({at_weight_tensors_[IN_Q_BIAS],
                                                at_weight_tensors_[IN_K_BIAS],
                                                at_weight_tensors_[IN_V_BIAS]},
                                               0)
                                        .to(torch::kInt32);

    for (auto idx : {IN_K_DEQSCALE,
                     IN_V_DEQSCALE,
                     IN_K_BIAS,
                     IN_V_BIAS,
                     IN_K_OFFSET,
                     IN_V_OFFSET,
                     IN_K_SCALE,
                     IN_V_SCALE}) {
      at_weight_tensors_[idx] = at_placeholder_;
    }

    at_weight_tensors_[IN_MLP_W2_BIAS] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_BIAS],
                    at_weight_tensors_[IN_MLP_W1_BIAS]},
                   0);

    at_weight_tensors_[IN_MLP_W2_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_DEQSCALE],
                    at_weight_tensors_[IN_MLP_W1_DEQSCALE]},
                   0)
            .to(torch::kFloat32);

    for (auto idx : {IN_MLP_W1_BIAS,
                     IN_MLP_W1_OFFSET,
                     IN_MLP_W1_SCALE,
                     IN_MLP_W1_DEQSCALE}) {
      at_weight_tensors_[idx] = at_placeholder_;
    }

    at_weight_tensors_[IN_Q_OFFSET] =
        at_weight_tensors_[IN_Q_OFFSET].to(torch::kInt8).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_OFFSET] =
        at_weight_tensors_[IN_ATTENTION_OUT_OFFSET]
            .to(torch::kInt8)
            .to(device_);
    at_weight_tensors_[IN_MLP_W2_OFFSET] =
        at_weight_tensors_[IN_MLP_W2_OFFSET].to(torch::kInt8).to(device_);

    if (rank_id_ != 0) {
      torch::Tensor original_tensor = at_weight_tensors_[IN_ATTENTION_OUT_BIAS];
      auto shape = original_tensor.sizes();
      auto dtype = original_tensor.dtype();
      auto device = original_tensor.device();

      at_weight_tensors_[IN_ATTENTION_OUT_BIAS] = torch::zeros(
          shape, torch::TensorOptions().dtype(dtype).device(device));
    }
  }

  auto new_q_weight = torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                                  at_weight_tensors_[IN_K_WEIGHT],
                                  at_weight_tensors_[IN_V_WEIGHT]},
                                 0)
                          .transpose(0, 1);

  at_weight_tensors_[IN_Q_WEIGHT] = at_npu::native::npu_format_cast(
      new_q_weight.contiguous(), ACL_FORMAT_FRACTAL_NZ);

  at_weight_tensors_[IN_ATTENTION_OUT_WEIGHT] = at_npu::native::npu_format_cast(
      at_weight_tensors_[IN_ATTENTION_OUT_WEIGHT].transpose(0, 1).contiguous(),
      ACL_FORMAT_FRACTAL_NZ);

  auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                                    at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                                   0)
                            .transpose(0, 1);

  at_weight_tensors_[IN_MLP_W2_WEIGHT] = at_npu::native::npu_format_cast(
      new_mlp_weight.contiguous(), ACL_FORMAT_FRACTAL_NZ);

  at_weight_tensors_[IN_MLP_CPROJ_WEIGHT] = at_npu::native::npu_format_cast(
      at_weight_tensors_[IN_MLP_CPROJ_WEIGHT].transpose(0, 1).contiguous(),
      ACL_FORMAT_FRACTAL_NZ);

  for (auto idx :
       {IN_MLP_W1_WEIGHT, IN_K_WEIGHT, IN_V_WEIGHT, IN_K_BIAS, IN_V_BIAS}) {
    at_weight_tensors_[idx] = at_placeholder_;
  }

  if (enableAddNorm_) {
    if (quantize_type_.compare("w8a8") == 0) {
      // quantize
      torch::ScalarType weight_fill_dtype = torch::kBFloat16;
      int64_t weight_attn_shape = at_weight_tensors_[IN_Q_WEIGHT].size(-1);
      int64_t weight_mlp_shape = at_weight_tensors_[IN_MLP_W2_WEIGHT].size(-1);
      at_weight_tensors_[IN_QKV_SCALE_FILL] = at_weight_tensors_[IN_Q_SCALE]
                                                  .repeat(weight_attn_shape)
                                                  .to(weight_fill_dtype);
      at_weight_tensors_[IN_MLP_SCALE_FILL] =
          at_weight_tensors_[IN_MLP_W2_SCALE]
              .repeat(weight_mlp_shape)
              .to(weight_fill_dtype);
      at_weight_tensors_[IN_QKV_OFFSET_FILL] = at_weight_tensors_[IN_Q_OFFSET]
                                                   .repeat(weight_attn_shape)
                                                   .to(weight_fill_dtype);
      at_weight_tensors_[IN_MLP_OFFSET_FILL] =
          at_weight_tensors_[IN_MLP_W2_OFFSET]
              .repeat(weight_mlp_shape)
              .to(weight_fill_dtype);
    } else {
      // bfloat16 or float16
      for (auto idx : {IN_QKV_SCALE_FILL,
                       IN_QKV_OFFSET_FILL,
                       IN_MLP_SCALE_FILL,
                       IN_MLP_OFFSET_FILL}) {
        at_weight_tensors_[idx] = at_placeholder_;
      }
    }
  }
}
}  // namespace layer
}  // namespace xllm