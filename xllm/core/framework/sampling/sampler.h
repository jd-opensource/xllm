/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <optional>
#include <vector>

#include "sampling_params.h"

namespace xllm {

class Sampler final {
 public:
  Sampler() = default;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // logits: [batch_size, vocab_size]
  SampleOutput forward(torch::Tensor& logits,
                       const SamplingParameters& params) const;

  // Set runtime-resolved global candidate token ids for local->global mapping.
  void set_candidate_token_ids(
      std::vector<int64_t> candidate_token_ids,
      std::optional<torch::Device> device = std::nullopt);

  // helper functions
  // probs: [..., vocab_size]
  static torch::Tensor greedy_sample(const torch::Tensor& probs);

  // probs: [..., vocab_size]
  static torch::Tensor random_sample(const torch::Tensor& probs);

  // Convert sampled local ids in candidate space back to global token ids.
  torch::Tensor map_local_to_global_token_ids(
      const torch::Tensor& token_ids) const;

  bool has_candidate_token_ids() const {
    return candidate_token_ids_tensor_.defined() &&
           candidate_token_ids_tensor_.numel() > 0;
  }

 private:
  torch::Tensor candidate_token_ids_tensor_;
};

}  // namespace xllm
