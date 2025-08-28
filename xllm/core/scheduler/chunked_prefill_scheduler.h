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

#include "scheduler/continuous_scheduler.h"

namespace xllm {

// TODO: Add Unit Testing later
//
// ContinuousScheduler using chunked prefill scheduling strategy
// Chunked-prefill: Sarathi-Serve https://arxiv.org/abs/2403.02310
// 1. we set the max tokens num be handled per batch.
// 2. Adhere to the "decode-maximal batching" principle:
//    Prioritize adding Decode requests to the batch until
//    the KV cache space they occupy reaches the upper limit.
//    Subsequently, based on the remaining token quota,
//    split sequences requiring Prefill into chunks and
//    add them to the batch.
class ChunkedPrefillScheduler final : public ContinuousScheduler {
 public:
  ChunkedPrefillScheduler(Engine* engine, const Options& options);
  virtual ~ChunkedPrefillScheduler();

  std::vector<Batch> prepare_batch_test() { return prepare_batch(); }

  uint32_t get_waiting_requests_num() const override {
    return waiting_priority_queue_.size();
  };

 private:
  // build a batch of requests from the priority queue
  virtual std::vector<Batch> prepare_batch() override;
  void handle_abnormal_request(
      const std::vector<Sequence*>& candidate_sequences,
      const std::vector<size_t>& candidate_token_budgets,
      const size_t& allocated_tokens,
      const size_t& allocated_seqs,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      bool budget_exhausted,
      bool block_exhausted);
  void handle_running_queue_requests(
      const size_t max_tokens_per_chunk_for_prefill,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      size_t& num_preempted_requests,
      std::vector<Sequence*>& prefill_stage_sequences,
      bool& budget_exhausted,
      bool& blocks_exhausted);
  void handle_prefill_requests(
      const size_t max_tokens_per_chunk_for_prefill,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<Sequence*>& prefill_stage_sequences,
      bool& blocks_exhausted,
      std::vector<std::shared_ptr<Request>>& finished_requests);
  void handle_remaining_budget(size_t& remaining_token_budget,
                               std::vector<Sequence*>& prefill_stage_sequences,
                               bool& blocks_exhausted);

  // 1. for prefill sequence: the allocated_tokens will be within
  // [1, num_prompt_tokens - num_tokens_in_kv_cache].
  // 2. for decode sequence: the allocated_tokens usually would
  // be 1 or K for speculative decoding.
  // returns false if no blocks can be allocated.
  bool allocate_blocks_for(Sequence* sequence,
                           size_t token_budget,
                           size_t* actual_tokens);

  void allocate_shared_blocks_for(Sequence* sequence);
};

}  // namespace xllm
