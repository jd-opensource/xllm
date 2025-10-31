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

#include <absl/time/time.h>
#include <folly/futures/Future.h>

#include <cstdint>
#include <vector>

#include "core/common/types.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/slice.h"
#include "finish_reason.h"
#include "framework/block/block.h"
#include "incremental_decoder.h"
#include "mm_data.h"
#include "request_output.h"
#include "sequence_kv_state.h"
#include "sequence_logprob_state.h"
#include "stopping_checker.h"

namespace xllm {

enum class SequenceStage : int8_t { PREFILL = 0, DECODE = 1 };

struct SequenceParams {
  // max tokens count in the sequence.
  size_t seq_capacity = 0;

  // whether to skip special tokens in the output text. default = true.
  bool skip_special_tokens = true;

  // whether to echo the prompt in the output text. default = false.
  bool echo = false;

  // whether to return the log probabilities of the tokens. default = false.
  bool logprobs = false;

  // the num of outputs.
  size_t n = 1;

  // the num of sequences to generate for each prompt and select the best
  // among.
  size_t best_of = 1;

  // whether the request is streaming
  bool streaming = false;

  // enable_schedule_overlap or not. default = false.
  bool enable_schedule_overlap = false;

  // whether this is a rec model. default = false.
  bool is_rec_model = false;
  int32_t bos_token_id = 0;
  // sampling params
  // reference from request
  RequestSamplingParam* sampling_param;  // not owned

  // stopping checker
  // reference from request
  StoppingChecker* stopping_checker;  // not owned
};

class Sequence final {
 public:
  Sequence(size_t index,
           const std::vector<int32_t>& prompt_token_ids,
           torch::Tensor input_embedding,
           const MMData& mm_data,
           const IncrementalDecoder& incremental_decoder,
           const SequenceParams& seq_params);

  Sequence(const Sequence& other);

  // get mm data
  const MMData& get_mm_data() const { return mm_data_; }
  void set_mrope_position_delta(int val) { mrope_position_delta_ = val; }
  int get_mrope_position_delta() { return mrope_position_delta_; }

  // get token ids to count map
  const std::unordered_map<int32_t, int32_t>& token_to_count_map() const {
    return token_to_count_map_;
  }

  // check if in prefill stage
  bool is_prefill_stage() const { return stage() == SequenceStage::PREFILL; }
  // get the sequence stage
  SequenceStage stage() const {
    if ((kv_state_.kv_cache_tokens_num() <
         std::max(volatile_num_prompt_tokens_, num_prompt_tokens())) &&
        kv_state_.kv_cache_tokens_num() > 0) {
      return SequenceStage::PREFILL;
    }
    return SequenceStage::DECODE;
  }

  // whether the new added token is the first token
  bool is_first_token() const { return is_first_token_; }
  std::optional<RemoteToken>& first_token() { return first_token_; }
  // get the total number of tokens
  size_t num_tokens() const { return num_tokens_; }
  // get the number of prompt tokens
  size_t num_prompt_tokens() const { return num_prompt_tokens_; }
  // get the number of generated tokens
  // returns 0 in prefill stage
  size_t num_generated_tokens() const {
    return num_tokens_ - num_prompt_tokens_;
  }
  Slice<int32_t> tokens() const { return {tokens_, num_tokens_}; }
  // get tokens in kv cache
  Slice<int32_t> cached_tokens() const {
    return {tokens_, kv_state_.kv_cache_tokens_num()};
  }

  // get token ids in host kv cache
  Slice<int32_t> cached_host_tokens() const {
    return {tokens_, host_kv_state_.kv_cache_tokens_num()};
  }

  // get the number of tokens need compute
  size_t num_need_compute_tokens() const {
    return num_tokens_ - std::max(kv_state_.kv_cache_tokens_num(),
                                  host_kv_state_.kv_cache_tokens_num());
  }

  // add a new token id to the sequence and update the count
  // the token would be discarded if the sequence is still in prefill stage
  void append_token(const Token& token);
  void append_token(int64_t token_id) { append_token(Token(token_id)); }
  void update_token(size_t index, const Token& token);
  void update_last_step_token(const Token& token, size_t token_offset = 0);
  bool has_new_tokens_generated() const {
    return num_tokens_ > decoder_.output_offset();
  }

  // update embeddings to the sequence
  void update_embeddings(const torch::Tensor& embedding);
  int32_t get_embedding_id() const { return embedding_id_; }
  // get input embedding
  torch::Tensor get_input_embedding() const { return input_embedding_; }

  void add_kv_blocks(const std::vector<Block>& blocks);
  void add_host_kv_blocks(const std::vector<Block>& blocks);
  void add_shared_kv_blocks(std::vector<Block>&& blocks);
  void add_shared_host_kv_blocks(std::vector<Block>&& blocks);

  // whether the prefill stage has been cached.
  bool if_cache_block_for_prefill() {
    bool if_cache =
        !is_cache_block_for_prefill_ && num_tokens() > num_prompt_tokens();
    is_cache_block_for_prefill_ |= if_cache;
    return if_cache;
  }

  FinishReason finish_reason() const { return finish_reason_; }
  // check finish status, use cached value if not invalidated
  bool finished() const;

  // get the output of the sequence until the specified number of tokens,
  // returns nullopt if no delta text and not finished
  std::optional<SequenceOutput> generate_streaming_output(
      size_t size,
      const Tokenizer& tokenizer);
  // get the full output of the sequence
  SequenceOutput generate_output(const Tokenizer& tokenizer);
  SequenceOutput generate_output();

  // get the sampling parameters
  const RequestSamplingParam* sampling_param() const {
    return sequence_params_.sampling_param;
  }

  // get the stopping criteria
  const StoppingChecker* stopping_checker() const {
    return sequence_params_.stopping_checker;
  }

  // close the sequence once all outputs have been sent
  void close() { closed_ = true; }
  bool is_closed() const { return closed_; }

  // finish the sequence by setting finished status and reason
  void finish();

  // time between two tokens
  int64_t tbt(const absl::Time& now);
  // set sequence ttft
  void set_time_to_first_token_latency_seconds(
      double time_to_first_token_latency_seconds) {
    time_to_first_token_latency_seconds_ = time_to_first_token_latency_seconds;
  }
  double time_to_first_token_latency_seconds() const {
    return time_to_first_token_latency_seconds_;
  }

  void set_dp_rank(int32_t dp_rank) { dp_rank_ = dp_rank; }

  int32_t dp_rank() const { return dp_rank_; }

  void enable_checking_prefill_token() {
    decoder_.enable_checking_prefill_token();
  }

  // get all generated token IDs (excluding prompt tokens)
  Slice<int32_t> get_generated_tokens() const;

  std::queue<bool>& pre_scheduled_step_prefill_queue() {
    return is_pre_scheduled_step_prefill_;
  }

  KVCacheState& kv_state() { return kv_state_; }

  KVCacheState& host_kv_state() { return host_kv_state_; }

  // for generated tokens
  float get_average_logprob();
  void generate_output_tokens_logprobs(
      size_t start_idx,
      size_t end_idx,
      const Tokenizer& tokenizer,
      std::optional<std::vector<LogProb>>& out_logprobs);

  void set_async_result(std::vector<folly::SemiFuture<uint32_t>>&& futures) {
    futures_ = std::move(futures);
  }

  void sync_result() {
    if (futures_.has_value()) {
      auto success_cnt = host_kv_state_.num_kv_blocks();
      for (auto& future : futures_.value()) {
        if (future.isReady()) {
          success_cnt = std::min(success_cnt, size_t(future.value()));
        } else {
          return;
        }
      }
      if (success_cnt > 0) {
        host_kv_state_.incr_kv_cache_tokens_num(
            success_cnt * host_kv_state_.kv_blocks()[0].size());
      }
    }
  }

  void reset();

  bool check_beam_search() {
    return sequence_params_.sampling_param->beam_width > 1;
  }

  LogprobState* logprob_state() { return logprob_state_.get(); }

  // set sequence id
  void set_seq_id(int32_t seq_id) { seq_id_ = seq_id; }

  // get sequence id
  int32_t seq_id() const { return seq_id_; }

  // rec model specific: get encoder tokens
  const std::vector<int32_t>& encoder_tokens() const { return encoder_tokens_; }

  // rec model specific: get encoder sequence length
  size_t encoder_seq_len() const { return num_encoder_tokens_; }

  // rec model specific: get number of decoder embeddings
  size_t num_decoder_embeddings() const { return num_decoder_embeddings_; }

  // rec model specific: check if this is a rec model
  bool is_rec_model() const { return is_rec_model_; }

  // rec model specific: static constants for embedding names
  static const std::string ENCODER_SPARSE_EMBEDDING_NAME;
  static const std::string DECODER_CONTEXT_EMBEDDING_NAME;

 private:
  // the index of the sequence in the request
  size_t index_ = 0;

  KVCacheState kv_state_;

  KVCacheState host_kv_state_;

  std::unique_ptr<LogprobState> logprob_state_;

  // latest token generate time
  absl::Time latest_generate_time_;

  // sequence ttft latency
  double time_to_first_token_latency_seconds_;

  // whether the added token is the first generated token
  bool is_first_token_ = false;

  // whether the prefill stage has been cached.
  bool is_cache_block_for_prefill_ = false;

  SequenceParams sequence_params_;

  // incremental decoder to decode the tokens
  IncrementalDecoder decoder_;

  // token ids generated for the sequence
  std::vector<int32_t> tokens_;

  torch::Tensor input_embedding_;

  MMData mm_data_;
  int mrope_position_delta_ = 0;

  // embeddings of the sequence
  torch::Tensor output_embedding_;

  // number of tokens in the sequence
  size_t num_tokens_ = 0;

  // the count of each token id
  std::unordered_map<int32_t, int32_t> token_to_count_map_;

  // the length of the prompt tokens
  size_t num_prompt_tokens_ = 0;

  // rec model specific: number of encoder tokens
  size_t num_encoder_tokens_ = 0;

  // rec model specific: number of decoder embeddings
  size_t num_decoder_embeddings_ = 0;

  // rec model specific: encoder tokens storage
  std::vector<int32_t> encoder_tokens_;

  // rec model specific: whether this is a rec model
  bool is_rec_model_ = false;

  // NOTE: MUST FIXME Later
  // record all tokens num in last turn when the request is
  // interrupted due to the lack of kv cache capacity.
  // All block tables are released when request be interrupted,
  // but the generated tokens are retained for the next execution.
  // In the next execution, we should treat these generated tokens as prompts.
  size_t volatile_num_prompt_tokens_ = 0;

  // embedding id that hold the embedding.
  int32_t embedding_id_ = -1;

  // is the sequence finished
  mutable bool finished_ = false;

  // is the finish status invalidated
  mutable bool finish_status_invalidated_ = true;

  // the reason why the sequence is finished
  mutable FinishReason finish_reason_ = FinishReason::NONE;

  // is the sequence closed.
  bool closed_ = false;

  // dp_rank
  int32_t dp_rank_ = -1;

  // seq id in the batch
  int32_t seq_id_ = -1;

  // for enable_schedule_overlap case
  uint32_t cur_generated_token_idx_;

  // record first token in disaggregated PD mode.
  std::optional<RemoteToken> first_token_;

  // when enable_schedule_overlap, use this to record whether is the
  // "prefill" stage at the execution of appending the fake token id.
  // In the appending stage, we push the state into the queue, in the
  // update stage, we pop the state from the queue.
  // 2 valid elements at most, maximum 2 steps pre scheduled.
  std::queue<bool> is_pre_scheduled_step_prefill_;

  // kvcache store copy async result
  std::optional<std::vector<folly::SemiFuture<uint32_t>>> futures_;
};

}  // namespace xllm
