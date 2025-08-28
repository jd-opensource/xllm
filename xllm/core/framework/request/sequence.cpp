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

#include "sequence.h"

#include <absl/strings/match.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/slice.h"
#include "core/util/tensor_helper.h"

namespace xllm {

Sequence::Sequence(size_t index,
                   const std::vector<int32_t>& prompt_token_ids,
                   torch::Tensor input_embedding,
                   const MMData& mm_data,
                   const IncrementalDecoder& decoder,
                   const SequenceParams& seq_params)
    : index_(index),
      mm_data_(mm_data),
      latest_generate_time_(absl::Now()),
      sequence_params_(seq_params),
      decoder_(std::move(decoder)) {
  CHECK(!prompt_token_ids.empty()) << "empty prompt token ids";
  auto capacity = sequence_params_.seq_capacity;
  CHECK_GT(capacity, prompt_token_ids.size()) << "capacity too small";

  num_prompt_tokens_ = prompt_token_ids.size();
  volatile_num_prompt_tokens_ = num_prompt_tokens_;
  tokens_.resize(capacity);

  // init logprob state
  logprob_state_ = std::make_unique<LogprobState>(num_prompt_tokens_, capacity);

  // add the prompt tokens
  for (const auto token_id : prompt_token_ids) {
    tokens_[num_tokens_++] = token_id;
    token_to_count_map_[token_id]++;
  }
  input_embedding_ = input_embedding;
  cur_generated_token_idx_ = num_prompt_tokens_;
}

void Sequence::append_token(const Token& token) {
  CHECK_LT(num_tokens_, tokens_.size())
      << "exceed the token capacity of the sequence";
  CHECK(!finished_) << "cannot append token to a finished sequence";
  CHECK(kv_state_.kv_cache_tokens_num() > 0 && !is_prefill_stage())
      << "cannot append token to a prefill sequence";

  if (!sequence_params_.enable_schedule_overlap) {
    // check if the token is the first token after the prompt
    is_first_token_ = num_tokens_ == num_prompt_tokens_;
  }

  // TODO: not record in non-disagg pd mode.
  if (is_first_token_) {
    RemoteToken t;
    t.token_id = token.id;
    if (token.logprob.has_value()) {
      t.token_logprob = token.logprob.value();
    }
    t.token_top_tokens = token.top_tokens;
    t.token_top_logprobs = token.top_logprobs;
    first_token_ = std::move(t);
  }

  // append the token id and update the token count
  const auto cur_idx = num_tokens_++;
  kv_state_.set_kv_cache_tokens_num(cur_idx);
  const int32_t token_id = static_cast<int32_t>(token.id);
  tokens_[cur_idx] = token_id;

  // skip update in enable_schedule_overlap
  if (sequence_params_.enable_schedule_overlap && token_id < 0) {
    finish_status_invalidated_ = true;
    return;
  }

  token_to_count_map_[token_id]++;
  // update logprobs if needed
  if (sequence_params_.sampling_param->logprobs) {
    logprob_state_->update_logprob(
        cur_idx, token, sequence_params_.sampling_param->top_logprobs);
  }

  // invalidate the finish status once a new token is appended
  finish_status_invalidated_ = true;
}

void Sequence::update_last_step_token(const Token& token, size_t token_offset) {
  if (sequence_params_.enable_schedule_overlap) {
    // check if the token is the first token after the prompt
    is_first_token_ = cur_generated_token_idx_ == num_prompt_tokens_;
  }

  // TODO: not record in non-disagg pd mode.
  if (is_first_token_) {
    RemoteToken t;
    t.token_id = token.id;
    if (token.logprob.has_value()) {
      t.token_logprob = token.logprob.value();
    }
    t.token_top_tokens = token.top_tokens;
    t.token_top_logprobs = token.top_logprobs;
    first_token_ = std::move(t);
  }

  // for mtp, currently only support multi-nodes task.
  if (token_offset > 0) {
    kv_state_.incr_kv_cache_tokens_num(1);
    num_tokens_++;
    // when enable speculative decoding, fake token id will be covered.
    tokens_[cur_generated_token_idx_ + 2] =
        tokens_[cur_generated_token_idx_ + 1];
    tokens_[cur_generated_token_idx_ + 1] = tokens_[cur_generated_token_idx_];
  }

  const int32_t token_id = static_cast<int32_t>(token.id);
  tokens_[cur_generated_token_idx_] = token_id;
  token_to_count_map_[token_id]++;
  // update logprobs if needed
  if (sequence_params_.sampling_param->logprobs) {
    logprob_state_->update_logprob(
        cur_generated_token_idx_,
        token,
        sequence_params_.sampling_param->top_logprobs);
  }
  ++cur_generated_token_idx_;
  finish_status_invalidated_ = true;
}

void Sequence::update_embeddings(const torch::Tensor& embeddings) {
  // cannot update embeddings to a finished sequence
  if (finished_) {
    return;
  }
  if (embeddings.defined()) {
    output_embedding_ = embeddings;
  }
  if (sequence_params_.sampling_param->is_embeddings) {
    // invalidate the finish status once a new token is appended
    finish_status_invalidated_ = false;
    finished_ = true;
    finish_reason_ = FinishReason::STOP;
  } else {
    if (output_embedding_.dim() == 1) {
      output_embedding_ = output_embedding_.unsqueeze(0);
    }
    mm_data_ = MMData(MMType::EMBEDDING, {{"embedding", output_embedding_}});
  }
}

std::optional<SequenceOutput> Sequence::generate_streaming_output(
    size_t size,
    const Tokenizer& tokenizer) {
  // figure out the valid generated token
  // because there might be fake token -1 if enable_schedule_overlap
  for (auto i = num_tokens_ - 1; i >= 0; --i) {
    if (tokens_[i] >= 0) {
      size = i + 1;
      break;
    }
  }
  CHECK_LE(size, num_tokens_);
  AUTO_COUNTER(detokenization_latency_seconds_stream);
  const auto ids = Slice<int32_t>(tokens_, size);

  // record the start index of token ids
  const size_t start = decoder_.output_offset();
  auto delta = decoder_.decode(ids, tokenizer);
  // NOTE:
  // There is a incomprehensible logic here: we use a thread pool to handle
  // request callbacks in response handler, which means that the main thread and
  // the tasks processing callbacks execute concurrently. This gives rise to a
  // scenario where the main thread finish forwarding, but the callbacks of some
  // previous steps have not yet been executed. However, the main thread
  // forwarding operation modifies sequence information such as um_tokens, which
  // may cause callback handle a previous step to process all accumulated tokens
  // directly when executing "generate_streaming_output" in a streaming
  // scenario. Example: output-1:
  // - step1: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":",
  //   I'm"}]}
  // - step2: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":",
  //   trying"}]}
  // - step3: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":",
  //   to"}]}
  // output-2:
  // - step1: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":",
  //   I'm trying to"}]}
  // - step2: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":"","finish_reason":"length"}]}
  // - step3: data:
  // {"id":"1","object":"text_completion","created":1,"model":"model","choices":[{"index":0,"text":"","finish_reason":"length"}]}
  //
  // We consider both of these cases to be valid,
  // subsequent callbacks only need to skip to return tokens.
  //
  if (delta.empty()) {
    return std::nullopt;
  }

  SequenceOutput output;
  output.index = index_;
  output.text = std::move(delta);

  const size_t end = decoder_.output_offset();
  output.token_ids = ids.slice(start, end);
  generate_output_tokens_logprobs(start, end, tokenizer, output.logprobs);

  return output;
}

SequenceOutput Sequence::generate_output() {
  SequenceOutput output;
  output.index = index_;
  if (finish_reason_ != FinishReason::NONE) {
    output.finish_reason = finish_reason_.to_string();
  }

  return output;
}

SequenceOutput Sequence::generate_output(const Tokenizer& tokenizer) {
  AUTO_COUNTER(detokenization_latency_seconds_non_stream);

  // build embeddings for output
  if (sequence_params_.sampling_param->is_embeddings) {
    SequenceOutput output;
    output.index = index_;
    Slice<float> embedding_slice = {output_embedding_.data_ptr<float>(),
                                    output_embedding_.size(0)};
    output.embeddings = embedding_slice;
    return output;
  }

  // NOTE: enable_schedule_overlap will generate an extra '-1' token.
  // we need to ignore these '-1' tokens.
  const auto ids = tokens();
  size_t size;
  for (auto i = num_tokens_ - 1; i >= 0; --i) {
    if (tokens_[i] >= 0) {
      size = i + 1;
      break;
    }
  }

  // record the start index of token ids
  const size_t start = decoder_.output_offset();

  // decide which position to start incremental decoding
  // leave 6 tokens for potential unfinished byte sequence
  size_t incremental_start = size <= 6 ? 0 : size - 6;
  // at least start from the first generated token
  if (incremental_start < num_prompt_tokens_) {
    incremental_start = num_prompt_tokens_;
  }
  // incrementally decode tokens between [incremental_start, size)
  std::stringstream ss;
  for (size_t end = incremental_start; end <= size; ++end) {
    ss << decoder_.decode(ids.slice(0, end), tokenizer);
  }

  SequenceOutput output;
  output.index = index_;
  output.text = ss.str();
  if (output_embedding_.defined()) {
    output.embedding = output_embedding_;
  }

  if (finish_reason_ != FinishReason::NONE) {
    output.finish_reason = finish_reason_.to_string();
  }

  const size_t end = decoder_.output_offset();
  output.token_ids = ids.slice(start, end);
  generate_output_tokens_logprobs(start, end, tokenizer, output.logprobs);

  return output;
}

void Sequence::add_kv_blocks(const std::vector<Block>& blocks) {
  kv_state_.add_kv_blocks(blocks);
  // use the last prefill block id as the embedding id
  if (embedding_id_ == -1) {
    embedding_id_ = blocks.back().id();
  }
}

// release all cache blocks
void Sequence::reset() {
  kv_state_.reset();
  volatile_num_prompt_tokens_ = num_tokens_;
}

void Sequence::add_shared_kv_blocks(std::vector<Block>&& blocks) {
  kv_state_.add_shared_kv_blocks(std::move(blocks), num_tokens_);
}

bool Sequence::finished() const {
  // return the cached finish status
  if (!finish_status_invalidated_) {
    return finished_;
  }

  // Embedding sequence never be finished until it updates its embeddings
  if (finish_status_invalidated_ &&
      sequence_params_.sampling_param->is_embeddings) {
    return false;
  }

  // reset the finish status invalidation flag
  finish_status_invalidated_ = false;

  auto finish_reason =
      sequence_params_.stopping_checker->check(tokens(), num_prompt_tokens_);
  if (finish_reason != FinishReason::NONE) {
    finish_reason_ = finish_reason;
    finished_ = true;
    return true;
  }
  return false;
}

int64_t Sequence::tbt(const absl::Time& now) {
  const int64_t latency =
      absl::ToInt64Milliseconds(now - latest_generate_time_);
  latest_generate_time_ = now;
  return latency;
}

float Sequence::get_average_logprob() {
  return logprob_state_->get_average_logprob(num_tokens_);
}

void Sequence::generate_output_tokens_logprobs(
    size_t start_idx,
    size_t end_idx,
    const Tokenizer& tokenizer,
    std::optional<std::vector<LogProb>>& out_logprobs) {
  if (!sequence_params_.logprobs || start_idx >= end_idx) {
    return;
  }

  logprob_state_->generate_output_tokens_logprobs(
      start_idx,
      end_idx,
      tokenizer,
      out_logprobs,
      sequence_params_.skip_special_tokens,
      tokens_);
}

}  // namespace xllm
