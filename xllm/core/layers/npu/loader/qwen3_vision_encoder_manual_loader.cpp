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

#include "qwen3_vision_encoder_manual_loader.h"

#include "qwen_loader_constants.h"

namespace xllm {
namespace layer {

using namespace qwen3_vision_encoder_constants;

Qwen3VisionEncoderManualLoader::Qwen3VisionEncoderManualLoader(
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
  dtype_ = torch::typeMetaToScalarType(options.dtype());
  for (int i = 0; i < weight_count; ++i) {
    at_host_weight_tensors_[i] = torch::zeros({1}, host_options);
  }
}

void Qwen3VisionEncoderManualLoader::load_state_dict(
    const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], true);
    } else {
      set_weight(state_dict, name, index, true);
    }
  }
}

void Qwen3VisionEncoderManualLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_host_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void Qwen3VisionEncoderManualLoader::merge_host_at_weights() {
  get_weights_col_packed_qkv();
  if (encode_param_world_size_ > 1) {
    at_host_weight_tensors_[IN_QKV_WEIGHT] =
        torch::cat({at_host_weight_tensors_[IN_VISION_Q_WEIGHT],
                    at_host_weight_tensors_[IN_VISION_K_WEIGHT],
                    at_host_weight_tensors_[IN_VISION_V_WEIGHT]},
                   0);
    at_host_weight_tensors_[IN_VISION_Q_WEIGHT] = torch::zeros({1});
    at_host_weight_tensors_[IN_VISION_K_WEIGHT] = torch::zeros({1});
    at_host_weight_tensors_[IN_VISION_V_WEIGHT] = torch::zeros({1});

    at_host_weight_tensors_[IN_QKV_BIAS] =
        torch::cat({at_host_weight_tensors_[IN_VISION_Q_BIAS],
                    at_host_weight_tensors_[IN_VISION_K_BIAS],
                    at_host_weight_tensors_[IN_VISION_V_BIAS]},
                   0);
    at_host_weight_tensors_[IN_VISION_Q_BIAS] = torch::zeros({1});
    at_host_weight_tensors_[IN_VISION_K_BIAS] = torch::zeros({1});
    at_host_weight_tensors_[IN_VISION_V_BIAS] = torch::zeros({1});
  }
}

void Qwen3VisionEncoderManualLoader::get_weights_col_packed_qkv() {
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

}  // namespace layer
}  // namespace xllm
