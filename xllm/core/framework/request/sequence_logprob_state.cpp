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

#include "sequence_logprob_state.h"

#include <absl/strings/match.h>

namespace xllm {

LogprobState::LogprobState(int64_t num_prompt_tokens, size_t capacity)
    : num_prompt_tokens_(num_prompt_tokens), acc_logprob_(0.0) {
  last_acc_token_idx_ = num_prompt_tokens_;
  logprobs_.resize(capacity);
  top_tokens_.resize(capacity);
  top_logprobs_.resize(capacity);
}

float LogprobState::get_average_logprob(int64_t num_tokens) {
  int64_t generated_tokens_num = num_tokens - num_prompt_tokens_;
  if (generated_tokens_num <= 0) {
    return std::numeric_limits<float>::min();
  }

  // no new tokens be generated ?
  if (num_tokens == last_acc_token_idx_) {
    return static_cast<float>(acc_logprob_ / generated_tokens_num);
  }

  CHECK(num_tokens > last_acc_token_idx_)
      << "num_tokens must be greater than last_acc_token_idx_, " << num_tokens
      << " vs " << last_acc_token_idx_;

  CHECK(last_acc_token_idx_ >= num_prompt_tokens_)
      << "last_acc_token_idx_ must be greater than or equal to "
         "num_prompt_tokens_, "
      << last_acc_token_idx_ << " vs " << num_prompt_tokens_;

  for (size_t i = last_acc_token_idx_; i < num_tokens; ++i) {
    if (logprobs_[i].has_value()) {
      acc_logprob_ += logprobs_[i].value();
    }
  }
  last_acc_token_idx_ = num_tokens;
  return static_cast<float>(acc_logprob_ / generated_tokens_num);
}

void LogprobState::generate_output_tokens_logprobs(
    size_t start_idx,
    size_t end_idx,
    const Tokenizer& tokenizer,
    std::optional<std::vector<LogProb>>& out_logprobs,
    bool skip_special_tokens,
    const std::vector<int32_t>& tokens) {
  if (start_idx < num_prompt_tokens_) {
    start_idx = num_prompt_tokens_;
  }

  for (size_t i = start_idx; i < end_idx; ++i) {
    if (!logprobs_[i].has_value()) {
      continue;
    }

    const int32_t token_id = tokens[i];
    auto token =
        tokenizer.decode(std::vector<int32_t>{token_id}, skip_special_tokens);
    if (token.empty()) {
      continue;
    }

    if (!out_logprobs.has_value()) {
      out_logprobs.emplace();
    }

    LogProb tmp_logprob;
    if (absl::EndsWith(token, "�")) {
      token = tokenizer.id_to_token(token_id);
      tmp_logprob.finished_token = false;
    }

    // add token and logprob
    tmp_logprob.token = std::move(token);
    tmp_logprob.token_id = token_id;
    tmp_logprob.logprob = logprobs_[i].value();

    // add top logprobs
    if (top_tokens_[i].empty()) {
      out_logprobs->emplace_back(std::move(tmp_logprob));
      continue;
    }

    const auto& top_tokens = top_tokens_[i];
    const auto& top_logprobs = top_logprobs_[i];
    DCHECK_EQ(top_tokens.size(), top_logprobs.size());
    std::vector<LogProbData> logprobs;
    for (size_t j = 0; j < top_tokens.size(); ++j) {
      LogProbData logprob;
      const int32_t top_token_id = top_tokens[j];
      const float top_logprob = top_logprobs[j];

      auto top_token = tokenizer.decode(std::vector<int32_t>{top_token_id},
                                        skip_special_tokens);
      if (absl::EndsWith(top_token, "�")) {
        top_token = tokenizer.id_to_token(top_token_id);
        logprob.finished_token = false;
      }

      logprob.token = top_token;
      logprob.token_id = top_token_id;
      logprob.logprob = top_logprob;
      logprobs.push_back(std::move(logprob));
    }
    tmp_logprob.top_logprobs = std::move(logprobs);

    out_logprobs->emplace_back(std::move(tmp_logprob));
  }
}

void LogprobState::update_logprob(size_t index,
                                  const Token& token,
                                  int64_t num_top_tokens) {
  CHECK(!logprobs_[index].has_value())
      << "logprob at index " << index << " is already set";
  logprobs_[index] = token.logprob;

  if (num_top_tokens > 0) {
    DCHECK_EQ(token.top_tokens.size(), token.top_logprobs.size());
    if (token.top_tokens.size() > num_top_tokens) {
      top_tokens_[index] = token.top_tokens.slice(0, num_top_tokens);
      top_logprobs_[index] = token.top_logprobs.slice(0, num_top_tokens);
    } else {
      DCHECK_EQ(token.top_tokens.size(), num_top_tokens);
      top_tokens_[index] = token.top_tokens;
      top_logprobs_[index] = token.top_logprobs;
    }
  }
}

}  // namespace xllm
