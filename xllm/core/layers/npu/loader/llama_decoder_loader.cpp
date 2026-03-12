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

#include "llama_decoder_loader.h"

#include "llama_loader_constants.h"

namespace xllm {
namespace layer {

using namespace llama_decoder_constants;

LlamaDecoderLoader::LlamaDecoderLoader(uint64_t weight_count,
                                       const ModelContext& context)
    : BaseLoader(weight_count, context) {
  at_weight_tensors_.resize(weight_count);

  auto options = context.get_tensor_options();
  dtype_ = torch::typeMetaToScalarType(options.dtype());

  for (int i = 0; i < weight_count; ++i) {
    at_weight_tensors_[i] = torch::zeros({1}).to(options);
  }
}

void LlamaDecoderLoader::verify_loaded_weights() const {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    CHECK(at_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void LlamaDecoderLoader::merge_loaded_weights() {
  auto new_q_weight = torch::cat({at_weight_tensors_[IN_Q_WEIGHT],
                                  at_weight_tensors_[IN_K_WEIGHT],
                                  at_weight_tensors_[IN_V_WEIGHT]},
                                 0);
  at_weight_tensors_[IN_Q_WEIGHT] = new_q_weight;

  at_weight_tensors_[IN_K_WEIGHT] = torch::zeros({1}).to(device_);
  at_weight_tensors_[IN_V_WEIGHT] = torch::zeros({1}).to(device_);

  auto new_mlp_weight = torch::cat({at_weight_tensors_[IN_MLP_W2_WEIGHT],
                                    at_weight_tensors_[IN_MLP_W1_WEIGHT]},
                                   0);
  at_weight_tensors_[IN_MLP_W2_WEIGHT] = new_mlp_weight;

  at_weight_tensors_[IN_MLP_W1_WEIGHT] = torch::zeros({1}).to(device_);
}

void LlamaDecoderLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index]);
    } else {
      set_weight(state_dict, name, index);
    }
  }
}

}  // namespace layer
}  // namespace xllm