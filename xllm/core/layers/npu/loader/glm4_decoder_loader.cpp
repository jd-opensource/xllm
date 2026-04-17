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

#include "glm4_decoder_loader.h"

#include "glm_loader_constants.h"

namespace xllm {
namespace layer {

using namespace glm4_decoder_constants;

Glm4DecoderLoader::Glm4DecoderLoader(uint64_t weight_count,
                                     const ModelContext& context)
    : BaseLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void Glm4DecoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

void Glm4DecoderLoader::merge_loaded_weights() {
  auto make_zero_like = [this](const torch::Tensor& ref) {
    return torch::zeros(
        {1}, torch::TensorOptions().dtype(ref.scalar_type()).device(device_));
  };

  at_weight_tensors_[IN_Q_WEIGHT] =
      torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                  at_weight_tensors_[IN_K_WEIGHT],
                  at_weight_tensors_[IN_V_WEIGHT]},
                 0)
          .contiguous();

  at_weight_tensors_[IN_Q_BIAS] = torch::cat({at_weight_tensors_[IN_Q_BIAS],
                                              at_weight_tensors_[IN_K_BIAS],
                                              at_weight_tensors_[IN_V_BIAS]},
                                             0)
                                      .contiguous();

  for (auto idx :
       {IN_MLP_W1_WEIGHT, IN_K_WEIGHT, IN_V_WEIGHT, IN_K_BIAS, IN_V_BIAS}) {
    at_weight_tensors_[idx] = make_zero_like(at_weight_tensors_[idx]);
  }
}

void Glm4DecoderLoader::verify_loaded_weights() const {
  for (const auto& [index, name] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

}  // namespace layer
}  // namespace xllm
