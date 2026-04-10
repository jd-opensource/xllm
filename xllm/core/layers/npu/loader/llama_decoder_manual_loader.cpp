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

#include "llama_decoder_manual_loader.h"

#include "llama_loader_constants.h"

namespace xllm {
namespace layer {

using namespace llama_decoder_constants;

LlamaDecoderManualLoader::LlamaDecoderManualLoader(uint64_t weight_count,
                                                   const ModelContext& context)
    : BaseManualLoader(weight_count, context) {
  auto options = context.get_tensor_options();
  auto host_options =
      torch::TensorOptions().dtype(options.dtype()).device(torch::kCPU);
  at_host_weight_tensors_.resize(weight_count);
  for (int i = 0; i < weight_count; ++i) {
    at_host_weight_tensors_[i] = torch::zeros({1}, host_options);
  }
}

void LlamaDecoderManualLoader::verify_loaded_weights() const {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    CHECK(at_host_weight_tensors_[index].sizes() != std::vector<int64_t>({1}))
        << "weight is not loaded for " << name;
  }
}

void LlamaDecoderManualLoader::merge_host_at_weights() {
  auto make_zero_like = [](const torch::Tensor& ref) {
    return torch::zeros(
        {1},
        torch::TensorOptions().dtype(ref.scalar_type()).device(torch::kCPU));
  };

  at_host_weight_tensors_[IN_Q_WEIGHT] =
      torch::cat({at_host_weight_tensors_[IN_Q_WEIGHT],
                  at_host_weight_tensors_[IN_K_WEIGHT],
                  at_host_weight_tensors_[IN_V_WEIGHT]},
                 0);

  at_host_weight_tensors_[IN_K_WEIGHT] =
      make_zero_like(at_host_weight_tensors_[IN_K_WEIGHT]);
  at_host_weight_tensors_[IN_V_WEIGHT] =
      make_zero_like(at_host_weight_tensors_[IN_V_WEIGHT]);

  at_host_weight_tensors_[IN_MLP_W2_WEIGHT] =
      torch::cat({at_host_weight_tensors_[IN_MLP_W2_WEIGHT],
                  at_host_weight_tensors_[IN_MLP_W1_WEIGHT]},
                 0);

  at_host_weight_tensors_[IN_MLP_W1_WEIGHT] =
      make_zero_like(at_host_weight_tensors_[IN_MLP_W1_WEIGHT]);
}

void LlamaDecoderManualLoader::load_state_dict(const StateDict& state_dict) {
  for (const auto& [name, index] : WEIGHT_MAPPING) {
    if (WEIGHT_SHARD.find(index) != WEIGHT_SHARD.end()) {
      set_weight(state_dict, name, index, WEIGHT_SHARD[index], true);
    } else {
      set_weight(state_dict, name, index, true);
    }
  }
}

}  // namespace layer
}  // namespace xllm
