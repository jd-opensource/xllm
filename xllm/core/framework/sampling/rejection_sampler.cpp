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

#include "rejection_sampler.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/stack.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <numeric>

#include "sampler.h"

namespace xllm {

namespace {
// index_select that supports multiple dimensions index
torch::Tensor index_select_2d(const torch::Tensor& input,
                              int64_t dim,
                              const torch::Tensor& index) {
  return input.gather(dim, index.unsqueeze(dim)).squeeze(dim);
}

}  // namespace

RejectionSamplerRateController::RejectionSamplerRateController(
    double fixed_acceptance_rate)
    : window_size_(1000),
      history_buffer_(1000, 0),
      history_idx_(0),
      window_sum_(0),
      error_buffer_(20, 0.0),
      error_idx_(0),
      pid_adj_(0.0),
      cumulative_err_(0.0),
      last_target_(-1.0),
      total_batches_(0),
      accepted_batches_(0),
      gen_(std::random_device{}()),
      dist_(0.0, 1.0),
      fixed_acceptance_rate_(fixed_acceptance_rate) {}

torch::Tensor RejectionSamplerRateController::FilterWithAcceptanceRate(
    const torch::Tensor& token_ids) {
  // Check parameters
  if (fixed_acceptance_rate_ < 0.0 || fixed_acceptance_rate_ > 1.0)
    return token_ids.clone();
  if (token_ids.size(0) == 0) return token_ids.clone();

  // Reset state if target acceptance rate changed
  if (std::abs(last_target_ - fixed_acceptance_rate_) > 1e-6) {
    ResetState(fixed_acceptance_rate_);
  }

  // Update window statistics
  total_batches_++;
  double global_rate =
      (total_batches_ > 0)
          ? static_cast<double>(accepted_batches_) / total_batches_
          : 0.0;
  window_sum_ -= history_buffer_[history_idx_];
  history_buffer_[history_idx_] = 0;
  double window_rate = static_cast<double>(window_sum_) / window_size_;

  // Calculate combined error between observed and target rates
  double batch_weight =
      1.0 - std::exp(-static_cast<double>(total_batches_) / 30.0);
  double win_weight =
      std::min(static_cast<double>(window_size_) / 100.0, 0.9) * batch_weight;
  double combined_err =
      (1.0 - win_weight) * (global_rate - fixed_acceptance_rate_) +
      win_weight * (window_rate - fixed_acceptance_rate_);

  // PID controller for automatic correction
  if (total_batches_ > 50) {
    double curr_err = global_rate - fixed_acceptance_rate_;
    error_buffer_[error_idx_] = curr_err;
    double prev_err = error_buffer_[(error_idx_ + 19) % 20];
    error_idx_ = (error_idx_ + 1) % 20;

    double i_term =
        std::accumulate(error_buffer_.begin(), error_buffer_.end(), 0.0);
    double d_term = curr_err - prev_err;

    // PID: kp=0.05, ki=0.001, kd=0.01
    pid_adj_ = 0.05 * curr_err + 0.001 * i_term + 0.01 * d_term;

    // Clamp adjustment
    double limit =
        0.02 +
        0.03 * (1.0 - std::exp(-static_cast<double>(total_batches_) / 500.0));
    pid_adj_ = std::clamp(pid_adj_, -limit, limit);
  }

  // Get corrected acceptance rate
  double adj_rate = CalculateAdjustedRate(fixed_acceptance_rate_, combined_err);

  // Decide accept or reject this batch
  bool accept = dist_(gen_) < adj_rate;

  torch::Tensor out_tensor = token_ids.clone();
  if (accept) {
    // Accept: keep as-is
    accepted_batches_++;
    history_buffer_[history_idx_] = 1;
    window_sum_ += 1;
  } else {
    // Reject: mask out tokens after the first (dimension 1)
    out_tensor.slice(1, 1).fill_(PLACEHOLDER_TOKEN_ID);
  }

  // Move window index forward
  history_idx_ = (history_idx_ + 1) % window_size_;

  // Update cumulative EMA error
  double actual_rate = static_cast<double>(accepted_batches_) / total_batches_;
  double alpha =
      0.05 * std::exp(-static_cast<double>(total_batches_) / 200.0) + 0.01;
  cumulative_err_ = alpha * (actual_rate - fixed_acceptance_rate_) +
                    (1.0 - alpha) * cumulative_err_;

  return out_tensor;
}

void RejectionSamplerRateController::ResetState(double new_rate) {
  pid_adj_ = 0.0;
  std::fill(error_buffer_.begin(), error_buffer_.end(), 0.0);
  last_target_ = new_rate;
}

double RejectionSamplerRateController::CalculateAdjustedRate(double target,
                                                             double error) {
  // 1. Nonlinear correction based on error magnitude
  double err_abs = std::abs(error);
  double factor = 1.0;
  if (err_abs > 0.0005) {
    double strength = 2.0 + (1.0 - std::exp(-err_abs * 50.0)) * 1.5;
    double sign = (error > 0) ? 1.0 : -1.0;
    factor = 1.0 + (strength * err_abs * sign);
  }

  // 2. Apply PID adjustment
  double rate_base = target * (factor == 0.0 ? 1.0 : 1.0 / factor);
  double rate = std::clamp(rate_base - pid_adj_, 0.0, 1.0);

  // 3. Edge cases and periodic force accept for low rate
  if (target == 0.0) return 0.0;
  if (target == 1.0) return 1.0;
  if (target > 0 && target < 0.05) {
    int period = static_cast<int>(1.0 / target);
    if (total_batches_ % period == 0) return 1.0;
  }

  // 4. Gap correction to enforce long-term target
  long target_accepted = std::llround(total_batches_ * target);
  long gap = target_accepted - accepted_batches_;
  long gap_threshold = std::max(1L, static_cast<long>(total_batches_ * 0.005));

  if (std::abs(gap) >= gap_threshold) {
    double gap_relative =
        static_cast<double>(std::abs(gap)) / std::max(1L, total_batches_);
    double importance = 1.0 - std::exp(-gap_relative * 50.0);
    double strength =
        0.2 + 0.8 * std::exp(-static_cast<double>(total_batches_) / 1000.0);
    double boost = importance * strength;

    if (gap > 0) {
      // Need to accept more
      rate = std::min(1.0, rate + (1.0 - rate) * boost);
    } else {
      // Need to reject more
      rate = std::max(0.0, rate * (1.0 - boost));
    }
  }

  // 5. Random noise to prevent local optima
  if (rate > 0.01 && rate < 0.99 && total_batches_ > 100) {
    double noise_amp =
        0.01 * std::exp(-static_cast<double>(total_batches_) / 500.0);
    double noise = (dist_(gen_) * 2.0 - 1.0) * noise_amp;
    rate = std::clamp(rate + noise, 0.0, 1.0);
  }

  return rate;
}

RejectionSampler::RejectionSampler(
    const torch::Tensor& do_sample,
    bool all_random_sample,
    bool all_greedy_sample,
    bool logprobs,
    int64_t max_top_logprobs,
    std::shared_ptr<RejectionSamplerRateController> rate_controller)
    : logprobs_(logprobs),
      max_top_logprobs_(max_top_logprobs),
      all_random_sample_(all_random_sample),
      all_greedy_sample_(all_greedy_sample),
      rate_controller_(rate_controller) {
  CHECK(do_sample.defined());
  // [batch_size, 1]
  do_sample_ = do_sample.unsqueeze_(/*dim=*/-1);
}

// draft_token_ids: [batch_size, n_speculative_tokens]
// draft_probs: [batch_size, n_speculative_tokens, vocab_size]
// target_logits: [batch_size, n_speculative_tokens + 1, vocab_size]
// bonus_token_ids: [batch_size, 1]
// returns accepted tokens. [batch_size, n_speculative_tokens + 1]
SampleOutput RejectionSampler::forward(const torch::Tensor& draft_token_ids,
                                       const torch::Tensor& draft_probs,
                                       const torch::Tensor& target_logits,
                                       const torch::Tensor& bonus_token_ids,
                                       bool mask_out_rejected_tokens) const {
  CHECK_EQ(draft_token_ids.size(0), do_sample_.size(0))
      << "batch size mismatch";
  DCHECK_EQ(draft_token_ids.size(1), draft_probs.size(1));
  // DCHECK_EQ(draft_probs.sizes(), target_probs.sizes());

  // [batch_size, n_speculative_tokens + 1, vocab_size] FloatTensor
  auto target_probs =
      torch::softmax(target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  // filter out probs for bonus tokens
  target_probs = target_probs.slice(
      /*dim=*/1, /*start=*/0, /*end=*/target_probs.size(1) - 1);

  // [batch_size, n_speculative_tokens + 1]
  torch::Tensor accepted_token_ids;
  torch::Tensor masked_accepted_token_ids;
  if (all_greedy_sample_) {
    std::tie(accepted_token_ids, masked_accepted_token_ids) =
        greedy_sample(draft_token_ids,
                      target_probs,
                      bonus_token_ids,
                      mask_out_rejected_tokens);
  } else if (all_random_sample_) {
    auto uniform_rand =
        torch::rand(draft_token_ids.sizes(), draft_probs.options());
    std::tie(accepted_token_ids, masked_accepted_token_ids) =
        random_sample(draft_token_ids,
                      draft_probs,
                      target_probs,
                      uniform_rand,
                      bonus_token_ids,
                      mask_out_rejected_tokens);
  } else {
    auto uniform_rand =
        torch::rand(draft_token_ids.sizes(), draft_probs.options());
    // mixed sample, sample both then choose based on do_sample_
    auto [random, masked_random] = random_sample(draft_token_ids,
                                                 draft_probs,
                                                 target_probs,
                                                 uniform_rand,
                                                 bonus_token_ids,
                                                 mask_out_rejected_tokens);
    auto [greedy, masked_greedy] = greedy_sample(draft_token_ids,
                                                 target_probs,
                                                 bonus_token_ids,
                                                 mask_out_rejected_tokens);
    accepted_token_ids = torch::where(do_sample_, random, greedy);
    if (mask_out_rejected_tokens) {
      masked_accepted_token_ids =
          torch::where(do_sample_, masked_random, masked_greedy);
    }
  }

  SampleOutput output;
  if (rate_controller_) {
    output.next_tokens =
        rate_controller_->FilterWithAcceptanceRate(accepted_token_ids);
  } else {
    output.next_tokens = mask_out_rejected_tokens ? masked_accepted_token_ids
                                                  : accepted_token_ids;
  }

  if (logprobs_) {
    // log_softmax is equivalent to log(softmax) but more numerically stable
    // [batch_size, n_speculative_tokens + 1, vocab_size]
    auto target_logprobs = torch::log_softmax(
        target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

    // select the logprobs for each sequence
    const auto selected_logprobs =
        index_select_2d(target_logprobs, /*dim=*/-1, accepted_token_ids);
    // output.probs = selected_probs;
    output.logprobs = selected_logprobs;

    if (max_top_logprobs_ > 0) {
      auto [values, indices] =
          target_logprobs.topk(max_top_logprobs_, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }
  return output;
}

// build mask from accepted matrix
// for example: [[1, 1, 0, 1],   ->   [[1, 1, 1, 0, 0],
//               [1, 0, 0, 0]]         [1, 1, 0, 0, 0]]
torch::Tensor RejectionSampler::build_accepted_mask(
    const torch::Tensor& accepted) {
  // build the mask for the first rejected token
  const auto batch_size = accepted.size(0);
  const auto n_tokens = accepted.size(1);

  // use LongTensor since argmax does not support bool
  auto accepted_int64 = accepted.to(torch::kInt64);
  auto bonus_mask = torch::zeros({batch_size, 1}, accepted_int64.options());
  auto combined_mask = torch::cat({accepted_int64, bonus_mask}, /*dim=*/-1);
  // [batch_size, 1]
  auto first_rejected_mask =
      (1 - combined_mask).argmax(/*dim=*/1, /*keepdim=*/true);

  // [1, n_speculative_tokens + 1]
  auto indices =
      torch::arange(n_tokens + 1, accepted.device()).unsqueeze(/*dim=*/0);
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_mask = indices <= first_rejected_mask;
  return accepted_mask;
}

std::tuple<torch::Tensor, torch::Tensor> RejectionSampler::random_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  auto selected_draft_probs =
      index_select_2d(draft_probs, /*dim=*/-1, draft_token_ids);
  auto selected_target_probs =
      index_select_2d(target_probs, /*dim=*/-1, draft_token_ids);

  // std::min(probs, 1.0) element-wise
  auto acceptance_probs = (selected_target_probs / selected_draft_probs);
  auto accepted = (uniform_rand < acceptance_probs);

  // construct recovered probs
  auto recovered_probs = (target_probs - draft_probs).clamp_min_(0);
  // a small value to avoid division by zero
  const auto epsilon = 1e-6f;
  auto sum = recovered_probs.sum(-1, /*keepdim=*/true).clamp_min_(epsilon);
  recovered_probs.div_(sum);

  // resample on the recovered probs
  torch::Tensor recovered_token_ids = Sampler::random_sample(recovered_probs);

  auto combined = torch::where(accepted, draft_token_ids, recovered_token_ids);
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_token_ids = torch::cat({combined, bonus_token_ids}, /*dim=*/-1);
  torch::Tensor masked_accepted_token_ids;
  if (mask_out_rejected_tokens) {
    // build the mask for the first rejected token
    auto accepted_mask = build_accepted_mask(accepted);
    // mask out the rejected tokens with -1
    masked_accepted_token_ids =
        torch::where(accepted_mask,
                     accepted_token_ids,
                     -torch::ones_like(accepted_token_ids));
  }
  return {accepted_token_ids, masked_accepted_token_ids};
}

std::tuple<torch::Tensor, torch::Tensor> RejectionSampler::greedy_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& target_probs,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  auto target_token_ids = Sampler::greedy_sample(target_probs);

  // mask out the rejected tokens with -1
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_token_ids =
      torch::cat({target_token_ids, bonus_token_ids}, /*dim=*/-1);
  torch::Tensor masked_accepted_token_ids;
  if (mask_out_rejected_tokens) {
    // [batch_size, n_speculative_tokens + 1]
    auto accepted = (target_token_ids == draft_token_ids);
    auto accepted_mask = build_accepted_mask(accepted);
    // mask out the rejected tokens with -1
    masked_accepted_token_ids =
        torch::where(accepted_mask,
                     accepted_token_ids,
                     -torch::ones_like(accepted_token_ids));
  }
  return {accepted_token_ids, masked_accepted_token_ids};
}

}  // namespace xllm
