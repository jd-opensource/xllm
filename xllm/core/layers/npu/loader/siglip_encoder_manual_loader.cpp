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

#include "siglip_encoder_manual_loader.h"

#include <set>

namespace xllm {
namespace layer {

SiglipEncoderUpManualLoader::SiglipEncoderUpManualLoader(
    const ModelContext& context)
    : BaseLoader(0, context) {
  options_ = context.get_tensor_options();
}

void SiglipEncoderUpManualLoader::load_state_dict(const StateDict& state_dict) {
  const std::set<std::string> key_names = {"layer_norm1.weight",
                                           "layer_norm1.bias",
                                           "self_attn.q_proj.weight",
                                           "self_attn.q_proj.bias",
                                           "self_attn.k_proj.weight",
                                           "self_attn.k_proj.bias",
                                           "self_attn.v_proj.weight",
                                           "self_attn.v_proj.bias"};

  for (const auto& [name, tensor] : state_dict) {
    if (key_names.find(name) == key_names.end()) {
      continue;
    }
    weights_map_[name] = tensor.to(options_);
  }
}

SiglipEncoderDownManualLoader::SiglipEncoderDownManualLoader(
    const ModelContext& context)
    : BaseLoader(0, context) {
  options_ = context.get_tensor_options();
}

void SiglipEncoderDownManualLoader::load_state_dict(
    const StateDict& state_dict) {
  const std::set<std::string> key_names = {"self_attn.out_proj.weight",
                                           "self_attn.out_proj.bias",
                                           "layer_norm2.weight",
                                           "layer_norm2.bias",
                                           "mlp.fc1.weight",
                                           "mlp.fc1.bias",
                                           "mlp.fc2.weight",
                                           "mlp.fc2.bias"};

  for (const auto& [name, tensor] : state_dict) {
    if (key_names.find(name) == key_names.end()) {
      continue;
    }
    weights_map_[name] = tensor.to(options_);
  }
}

}  // namespace layer
}  // namespace xllm
