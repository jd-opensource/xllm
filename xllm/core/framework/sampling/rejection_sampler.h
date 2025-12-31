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

// Provide a default placeholder token ID
#ifndef PLACEHOLDER_TOKEN_ID
#define PLACEHOLDER_TOKEN_ID -1
#endif

class RejectionSamplerRateController {
 public:
  explicit RejectionSamplerRateController(double fixed_acceptance_rate);

  // Core filtering function, decides whether to accept a batch based on target
  // acceptance rate
  torch::Tensor filter_with_acceptance_rate(const torch::Tensor& token_ids);

 private:
  // Reset internal state (call when the target acceptance rate changes
  // significantly)
  void reset_state(double new_rate);

  // Compute the final acceptance rate after PID and error correction
  double calculate_adjusted_rate(double target, double error);
  size_t window_size_;

  // History state (using circular buffer logic)
  std::vector<int> history_buffer_;
  size_t history_idx_;
  long window_sum_;

  // PID controller state
  std::vector<double> error_buffer_;
  size_t error_idx_;
  double pid_adj_;

  // Global statistics and error state
  double cumulative_err_;
  double last_target_;
  long total_batches_;
  long accepted_batches_;

  // Random number generator
  std::mt19937 gen_;
  std::uniform_real_distribution<double> dist_;

  // acceptance rate
  double fixed_acceptance_rate_;
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
