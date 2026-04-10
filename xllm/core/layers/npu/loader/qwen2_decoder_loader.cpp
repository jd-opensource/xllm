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

#include "qwen2_decoder_loader.h"

#include "qwen_loader_constants.h"

namespace xllm {
namespace layer {

using namespace qwen2_decoder_constants;

Qwen2DecoderLoader::Qwen2DecoderLoader(uint64_t weight_count,
                                       const ModelContext& context)
    : BaseLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  device_id_ = options.device().index();

  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Qwen2DecoderLoader::load_state_dict(const StateDict& state_dict) {
  if (quantize_type_ == "w8a8") {
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

void Qwen2DecoderLoader::merge_loaded_weights() {
  if (quantize_type_ == "w8a8") {
    at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE] =
        at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
    at_weight_tensors_[IN_Q_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_Q_DEQSCALE],
                    at_weight_tensors_[IN_K_DEQSCALE],
                    at_weight_tensors_[IN_V_DEQSCALE]},
                   0)
            .to(torch::kFloat32);
    at_weight_tensors_[IN_K_DEQSCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_V_DEQSCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_K_OFFSET] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_V_OFFSET] = torch::zeros({1}).to(device_);

    at_weight_tensors_[IN_K_SCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_V_SCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_MLP_W2_BIAS] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_BIAS],
                    at_weight_tensors_[IN_MLP_W1_BIAS]},
                   0);
    at_weight_tensors_[IN_MLP_W1_BIAS] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_MLP_W2_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_DEQSCALE],
                    at_weight_tensors_[IN_MLP_W1_DEQSCALE]},
                   0)
            .to(torch::kFloat32);
    at_weight_tensors_[IN_MLP_W1_DEQSCALE] = torch::zeros({1}).to(device_);

    at_weight_tensors_[IN_MLP_W1_OFFSET] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_MLP_W1_SCALE] = torch::zeros({1}).to(device_);
    at_weight_tensors_[IN_Q_OFFSET] =
        at_weight_tensors_[IN_Q_OFFSET].to(torch::kInt8).to(device_);
    at_weight_tensors_[IN_ATTENTION_OUT_OFFSET] =
        at_weight_tensors_[IN_ATTENTION_OUT_OFFSET]
            .to(torch::kInt8)
            .to(device_);
    at_weight_tensors_[IN_MLP_W2_OFFSET] =
        at_weight_tensors_[IN_MLP_W2_OFFSET].to(torch::kInt8).to(device_);
    if (device_id_ != 0) {
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

  at_weight_tensors_[IN_K_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_V_WEIGHT] = torch::zeros({1}).to(device_);

  auto new_q_bias = torch::cat({at_weight_tensors_[IN_Q_BIAS],
                                at_weight_tensors_[IN_K_BIAS],
                                at_weight_tensors_[IN_V_BIAS]},
                               0);
  at_weight_tensors_[IN_Q_BIAS] = new_q_bias;

  at_weight_tensors_[IN_K_BIAS] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_V_BIAS] = torch::zeros({1}).to(device_);

  at_weight_tensors_[IN_ATTENTION_OUT_WEIGHT] = at_npu::native::npu_format_cast(
      at_weight_tensors_[IN_ATTENTION_OUT_WEIGHT].transpose(0, 1).contiguous(),
      ACL_FORMAT_FRACTAL_NZ);

  auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                                    at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                                   0)
                            .transpose(0, 1);

  at_weight_tensors_[IN_MLP_W2_WEIGHT] = at_npu::native::npu_format_cast(
      new_mlp_weight.contiguous(), ACL_FORMAT_FRACTAL_NZ);

  at_weight_tensors_[IN_MLP_W1_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_MLP_CPROJ_WEIGHT] = at_npu::native::npu_format_cast(
      at_weight_tensors_[IN_MLP_CPROJ_WEIGHT].transpose(0, 1).contiguous(),
      ACL_FORMAT_FRACTAL_NZ);
}

void Qwen2DecoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

}  // namespace layer
}  // namespace xllm