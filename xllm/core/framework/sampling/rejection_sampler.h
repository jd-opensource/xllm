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
#include <torch/torch.h>
#include <torch/types.h>

#include <random>

#include "sampling_params.h"

namespace xllm {

class RejectionSamplerRateController {
 public:
  explicit RejectionSamplerRateController(double fixed_acceptance_rate);

  // Core filtering function.
  // Maintains the target acceptance rate using a cumulative drift correction.
  torch::Tensor filter_with_acceptance_rate(const torch::Tensor& token_ids);

 private:
  // Placeholder value for rejected tokens
  static constexpr int32_t kPlaceholderTokenId = -1;

  // Tolerance for detecting when the user changes the target rate
  static constexpr double kTargetRateChangeTolerance = 1e-6;

  // Gain factor for drift correction (higher = faster correction, lower =
  // smoother)
  static constexpr double kDriftCorrectionGain = 0.1;

  // Target configuration
  double fixed_acceptance_rate_;
  double last_target_;

  // Global statistics for long-term rate tracking
  int64_t total_batches_ = 0;
  int64_t accepted_batches_ = 0;

  // Random number generator components
  std::mt19937 gen_{std::random_device{}()};
  std::uniform_real_distribution<double> dist_{0.0, 1.0};
};

class RejectionSampler final {
 public:
  RejectionSampler(const torch::Tensor& do_sample,
                   bool all_random_sample,
                   bool all_greedy_sample,
                   bool logprobs,
                   int64_t max_top_logprobs,
                   std::shared_ptr<RejectionSamplerRateController>
                       rate_controller = nullptr);

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // Sample tokens ids using rejection sampling.
  // draft_token_ids: [batch_size, n_speculative_tokens]
  // draft_probs: [batch_size, n_speculative_tokens, vocab_size]
  // target_logits: [batch_size, n_speculative_tokens + 1, vocab_size]
  // bonus_token_ids: [batch_size, 1]
  SampleOutput forward(const torch::Tensor& draft_token_ids,
                       const torch::Tensor& draft_probs,
                       const torch::Tensor& target_logits,
                       const torch::Tensor& bonus_token_ids,
                       bool mask_out_rejected_tokens = false) const;

  // build mask from accepted matrix
  // for example: [[1, 1, 0, 1],   ->   [[1, 1, 1, 0, 0],
  //               [1, 0, 0, 0]]         [1, 1, 0, 0, 0]]
  static torch::Tensor build_accepted_mask(const torch::Tensor& accepted);

  static std::tuple<torch::Tensor, torch::Tensor> random_sample(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& draft_probs,
      const torch::Tensor& target_probs,
      const torch::Tensor& uniform_rand,
      const torch::Tensor& bonus_token_ids,
      bool mask_out_rejected_tokens);

  static std::tuple<torch::Tensor, torch::Tensor> greedy_sample(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& target_probs,
      const torch::Tensor& bonus_token_ids,
      bool mask_out_rejected_tokens);

 private:
  // whether to return logprobs
  bool logprobs_ = false;

  // max number of top logprobs in the batch
  int64_t max_top_logprobs_ = 0;

  // [batch_size]
  torch::Tensor do_sample_;
  bool all_random_sample_ = true;
  bool all_greedy_sample_ = true;

  // rate controller for fixing the speculative acceptance rate
  std::shared_ptr<RejectionSamplerRateController> rate_controller_;
};

}  // namespace xllm
