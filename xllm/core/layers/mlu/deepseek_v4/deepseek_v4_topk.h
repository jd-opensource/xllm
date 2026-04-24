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

#pragma once

#include <torch/torch.h>

#include <optional>

#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

// Output struct for DeepSeekV4 TopK routing results.
struct DeepSeekV4TopKOutput {
  torch::Tensor weights;  // [batch_size, n_activated_experts] routing weights.
  torch::Tensor indices;  // [batch_size, n_activated_experts] expert indices.
};

// DeepSeekV4 TopK routing module.
// Supports sqrtsoftplus scoring function with optional hash-based routing.
// This module is specifically for sqrtsoftplus scoring function, as the fused
// MoE kernel does not support softplus and hash-based lookup.
class DeepSeekV4TopKImpl final : public torch::nn::Module {
 public:
  DeepSeekV4TopKImpl() = default;

  DeepSeekV4TopKImpl(int64_t n_routed_experts,
                     int64_t n_activated_experts,
                     float route_scale,
                     int64_t vocab_size,
                     bool use_hash,
                     const torch::TensorOptions& options);

  DeepSeekV4TopKOutput forward(
      const torch::Tensor& scores,
      const std::optional<torch::Tensor>& input_ids = std::nullopt);

  void load_state_dict(const StateDict& state_dict);

 private:
  int64_t n_routed_experts_ = 0;
  int64_t n_activated_experts_ = 0;
  float route_scale_ = 1.0f;
  int64_t vocab_size_ = 0;
  bool use_hash_ = false;

  // Hash-based routing: [vocab_size, n_activated_experts] token-to-expert map.
  DEFINE_WEIGHT(tid2eid);

  // Standard topk routing: [n_routed_experts] score correction bias.
  DEFINE_WEIGHT(bias);
};
TORCH_MODULE(DeepSeekV4TopK);

}  // namespace layer
}  // namespace xllm
