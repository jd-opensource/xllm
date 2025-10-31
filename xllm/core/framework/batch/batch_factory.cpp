/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "batch_factory.h"

namespace xllm {

namespace {

bool is_beam_search(const std::vector<std::shared_ptr<Request>>& requests) {
  for (const auto& request : requests) {
    if (request->check_beam_search()) {
      return true;
    }
  }
  return false;
}
}  // namespace

std::vector<Batch> BatchFactory::create_batches(
    const std::vector<std::shared_ptr<Request>>& running_requests,
    const std::vector<Sequence*>& running_sequences,
    const std::vector<size_t>& running_sequences_budgets,
    std::vector<std::vector<CacheBlockInfo>>* copy_in_cache_block_infos,
    std::vector<std::vector<CacheBlockInfo>>* copy_out_cache_block_infos,
    std::vector<std::vector<CacheBlockInfo>>* swap_cache_block_infos) {
  size_t num_prompt_tokens = 0;
  size_t num_generated_tokens = 0;
  std::vector<Batch> batches(dp_size_);
  for (size_t i = 0; i < running_sequences.size(); ++i) {
    auto* sequence = running_sequences[i];
    const size_t token_budget = running_sequences_budgets[i];

    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() >
                sequence->kv_state().kv_cache_tokens_num()
            ? sequence->num_prompt_tokens() -
                  sequence->kv_state().kv_cache_tokens_num()
            : 0;
    const size_t prompt_tokens =
        std::min(remaining_prompt_tokens, token_budget);
    const size_t generated_tokens = token_budget - prompt_tokens;
    num_prompt_tokens += prompt_tokens;
    num_generated_tokens += generated_tokens;

    // if dp enabled, each sequence is required to
    // dispatch to the same rank in the whole lifetime
    batches[sequence->dp_rank()].add(sequence, token_budget);
    if (!((sequence->stage() == SequenceStage::DECODE) &&
          (sequence->kv_state().kv_cache_tokens_num() > 0))) {
      batches[sequence->dp_rank()].set_batch_prefill_status(true);
    }
  }

  if (is_beam_search(running_requests)) {
    for (const auto& request : running_requests) {
      auto seq_group = request->sequence_group();
      int32_t dp_rank = seq_group->dp_rank();
      batches[dp_rank].add(seq_group);
    }
  }

  for (int i = 0; i < dp_size_; i++) {
    if (!batches[i].empty()) {
      if (copy_in_cache_block_infos != nullptr &&
          copy_in_cache_block_infos->size() == dp_size_) {
        batches[i].set_copy_in_cache_block_infos(
            &(copy_in_cache_block_infos->at(i)));
      }
      if (copy_out_cache_block_infos != nullptr &&
          copy_out_cache_block_infos->size() == dp_size_) {
        batches[i].set_copy_out_cache_block_infos(
            &(copy_out_cache_block_infos->at(i)));
      }
      if (swap_cache_block_infos != nullptr &&
          swap_cache_block_infos->size() == dp_size_) {
        batches[i].set_swap_cache_block_infos(&(swap_cache_block_infos->at(i)));
      }
    }
  }

  COUNTER_ADD(num_processing_tokens_total_prompt, num_prompt_tokens);
  COUNTER_ADD(num_processing_tokens_total_generated, num_generated_tokens);

  if (running_sequences.size() > 0) {
    HISTOGRAM_OBSERVE(
        num_prompt_tokens_per_request,
        static_cast<int64_t>(num_prompt_tokens / running_sequences.size()));
    HISTOGRAM_OBSERVE(
        num_generated_tokens_per_request,
        static_cast<int64_t>(num_generated_tokens / running_sequences.size()));
  }

  return batches;
}

std::vector<Batch> BatchFactory::create_rec_batches(
    const std::vector<std::shared_ptr<Request>>& running_requests,
    const std::vector<Sequence*>& running_sequences,
    const std::vector<size_t>& running_sequences_budgets,
    std::vector<std::vector<CacheBlockInfo>>* copy_in_cache_block_infos,
    std::vector<std::vector<CacheBlockInfo>>* copy_out_cache_block_infos,
    std::vector<std::vector<CacheBlockInfo>>* swap_cache_block_infos) {
  size_t num_prompt_tokens = 0;
  size_t num_generated_tokens = 0;
  std::vector<Batch> batches(dp_size_);
  for (size_t i = 0; i < running_sequences.size(); ++i) {
    auto* sequence = running_sequences[i];
    const size_t token_budget = running_sequences_budgets[i];

    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() >
                sequence->kv_state().kv_cache_tokens_num()
            ? sequence->num_prompt_tokens() -
                  sequence->kv_state().kv_cache_tokens_num()
            : 0;
    const size_t prompt_tokens =
        std::min(remaining_prompt_tokens, token_budget);
    const size_t generated_tokens = token_budget - prompt_tokens;
    num_prompt_tokens += prompt_tokens;
    num_generated_tokens += generated_tokens;

    // if dp enabled, each sequence is required to
    // dispatch to the same rank in the whole lifetime
    // batches[sequence->dp_rank()].add(sequence, token_budget);
    if (!((sequence->stage() == SequenceStage::DECODE) &&
          (sequence->kv_state().kv_cache_tokens_num() > 0))) {
      batches[sequence->dp_rank()].set_batch_prefill_status(true);
    }
  }
  // for rec, only use seq_group to prepare_input.
  for (const auto& request : running_requests) {
    auto seq_group = request->sequence_group();
    int32_t dp_rank = seq_group->dp_rank();
    batches[dp_rank].add(seq_group);
  }

  for (int i = 0; i < dp_size_; i++) {
    if (!batches[i].empty()) {
      if (copy_in_cache_block_infos != nullptr &&
          copy_in_cache_block_infos->size() == dp_size_) {
        batches[i].set_copy_in_cache_block_infos(
            &(copy_in_cache_block_infos->at(i)));
      }
      if (copy_out_cache_block_infos != nullptr &&
          copy_out_cache_block_infos->size() == dp_size_) {
        batches[i].set_copy_out_cache_block_infos(
            &(copy_out_cache_block_infos->at(i)));
      }
      if (swap_cache_block_infos != nullptr &&
          swap_cache_block_infos->size() == dp_size_) {
        batches[i].set_swap_cache_block_infos(&(swap_cache_block_infos->at(i)));
      }
    }
  }

  COUNTER_ADD(num_processing_tokens_total_prompt, num_prompt_tokens);
  COUNTER_ADD(num_processing_tokens_total_generated, num_generated_tokens);

  if (running_sequences.size() > 0) {
    HISTOGRAM_OBSERVE(
        num_prompt_tokens_per_request,
        static_cast<int64_t>(num_prompt_tokens / running_sequences.size()));
    HISTOGRAM_OBSERVE(
        num_generated_tokens_per_request,
        static_cast<int64_t>(num_generated_tokens / running_sequences.size()));
  }

  return batches;
}

}  // namespace xllm
