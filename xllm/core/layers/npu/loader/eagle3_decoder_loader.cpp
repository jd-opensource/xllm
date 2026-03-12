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

#include "eagle3_decoder_loader.h"

#include <set>

#include "eagle3_loader_constants.h"

namespace xllm {
namespace layer {

using namespace eagle3_decoder_constants;

Eagle3DecoderLoader::Eagle3DecoderLoader(uint64_t weight_count,
                                         const ModelContext& context)
    : BaseLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  device_id_ = options.device().index();

  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Eagle3DecoderLoader::load_state_dict(const StateDict& state_dict) {
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

    at_weight_tensors_[IN_HIDDEN_NORM_BIAS] =
        torch::zeros(at_weight_tensors_[IN_HIDDEN_NORM_WEIGHT].sizes(),
                     at_weight_tensors_[IN_HIDDEN_NORM_WEIGHT].options())
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

void Eagle3DecoderLoader::merge_loaded_weights() {
  auto make_zero_like = [this](const torch::Tensor& ref) {
    return torch::zeros(
        {1}, torch::TensorOptions().dtype(ref.scalar_type()).device(device_));
  };

  if (quantize_type_ == "w8a8") {
    at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE] =
        at_weight_tensors_[IN_ATTENTION_OUT_DEQSCALE].to(torch::kFloat32);
    at_weight_tensors_[IN_Q_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_Q_DEQSCALE],
                    at_weight_tensors_[IN_K_DEQSCALE],
                    at_weight_tensors_[IN_V_DEQSCALE]},
                   0)
            .to(torch::kFloat32);
    at_weight_tensors_[IN_K_DEQSCALE] =
        make_zero_like(at_weight_tensors_[IN_K_DEQSCALE]);
    at_weight_tensors_[IN_V_DEQSCALE] =
        make_zero_like(at_weight_tensors_[IN_V_DEQSCALE]);
    at_weight_tensors_[IN_K_OFFSET] =
        make_zero_like(at_weight_tensors_[IN_K_OFFSET]);
    at_weight_tensors_[IN_V_OFFSET] =
        make_zero_like(at_weight_tensors_[IN_V_OFFSET]);
    at_weight_tensors_[IN_K_SCALE] =
        make_zero_like(at_weight_tensors_[IN_K_SCALE]);
    at_weight_tensors_[IN_V_SCALE] =
        make_zero_like(at_weight_tensors_[IN_V_SCALE]);
    at_weight_tensors_[IN_MLP_W2_BIAS] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_BIAS],
                    at_weight_tensors_[IN_MLP_W1_BIAS]},
                   0);
    at_weight_tensors_[IN_MLP_W1_BIAS] =
        make_zero_like(at_weight_tensors_[IN_MLP_W1_BIAS]);
    at_weight_tensors_[IN_MLP_W2_DEQSCALE] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_DEQSCALE],
                    at_weight_tensors_[IN_MLP_W1_DEQSCALE]},
                   0)
            .to(torch::kFloat32);
    at_weight_tensors_[IN_MLP_W1_DEQSCALE] =
        make_zero_like(at_weight_tensors_[IN_MLP_W1_DEQSCALE]);
    at_weight_tensors_[IN_MLP_W1_OFFSET] =
        make_zero_like(at_weight_tensors_[IN_MLP_W1_OFFSET]);
    at_weight_tensors_[IN_MLP_W1_SCALE] =
        make_zero_like(at_weight_tensors_[IN_MLP_W1_SCALE]);
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

  at_weight_tensors_[IN_Q_WEIGHT] =
      torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                  at_weight_tensors_[IN_K_WEIGHT],
                  at_weight_tensors_[IN_V_WEIGHT]},
                 0)
          .contiguous();

  at_weight_tensors_[IN_K_WEIGHT] =
      make_zero_like(at_weight_tensors_[IN_K_WEIGHT]);
  at_weight_tensors_[IN_V_WEIGHT] =
      make_zero_like(at_weight_tensors_[IN_V_WEIGHT]);

  if (at_weight_tensors_[IN_Q_BIAS].sizes() != std::vector<int64_t>({1})) {
    at_weight_tensors_[IN_Q_BIAS] = torch::cat({at_weight_tensors_[IN_Q_BIAS],
                                                at_weight_tensors_[IN_K_BIAS],
                                                at_weight_tensors_[IN_V_BIAS]},
                                               0)
                                        .contiguous();

    at_weight_tensors_[IN_K_BIAS] =
        make_zero_like(at_weight_tensors_[IN_K_BIAS]);
    at_weight_tensors_[IN_V_BIAS] =
        make_zero_like(at_weight_tensors_[IN_V_BIAS]);
  }

  TransposeType transpose_type =
      check_transpose(at_weight_tensors_[IN_MLP_W2_WEIGHT]);
  if (transpose_type == TransposeType::TRANSPOSE) {
    at_weight_tensors_[IN_MLP_W2_WEIGHT] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                    at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                   0)
            .contiguous();
  } else {
    at_weight_tensors_[IN_MLP_W2_WEIGHT] =
        torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                    at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                   0)
            .transpose(0, 1)
            .contiguous();
  }

  at_weight_tensors_[IN_MLP_W1_WEIGHT] =
      make_zero_like(at_weight_tensors_[IN_MLP_W1_WEIGHT]);
}

TransposeType Eagle3DecoderLoader::check_transpose(at::Tensor& tensor) {
  bool is_k_divisible = tensor.size(1) % 256 == 0;
  bool is_n_divisible = tensor.size(0) % 256 == 0;

  if (!is_k_divisible && is_n_divisible) {
    return TransposeType::NOT_TRANSPOSE;
  }

  return TransposeType::TRANSPOSE;
}

void Eagle3DecoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

}  // namespace layer
}  // namespace xllm
