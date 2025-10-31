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

#include "sequences_group.h"

#include <unordered_set>

#include "framework/batch/beam_search.h"
#include "util/blocking_counter.h"

namespace xllm {

SequencesGroup::SequencesGroup(const std::string& prompt,
                               const std::vector<int32_t>& prompt_tokens,
                               const torch::Tensor& input_embedding,
                               const MMData& mm_data,
                               const SequenceParams& sequence_params)
    : prompt_(prompt),
      prompt_tokens_(prompt_tokens),
      input_embedding_(input_embedding),
      mm_data_(mm_data),
      sequence_params_(std::move(sequence_params)) {
  add();
}

void SequencesGroup::add() {
  const size_t index = sequences_.size();
  IncrementalDecoder decoder(prompt_,
                             prompt_tokens_.size(),
                             sequence_params_.echo,
                             sequence_params_.skip_special_tokens);
  sequences_.emplace_back(std::make_unique<Sequence>(index,
                                                     prompt_tokens_,
                                                     input_embedding_,
                                                     mm_data_,
                                                     std::move(decoder),
                                                     sequence_params_));
}

bool SequencesGroup::finished() const {
  if (sequences_.size() < sequence_params_.best_of) {
    return false;
  }

  return std::all_of(sequences_.begin(),
                     sequences_.end(),
                     [](const auto& sequence) { return sequence->finished(); });
}

bool SequencesGroup::expand_sequences(bool share_prefix) {
  // when enable beam search, dont need to expand sequences
  if (check_beam_search()) {
    return false;
  }
  size_t current_seq_count = sequences_.size();
  size_t best_of = sequence_params_.best_of;
  if (current_seq_count == best_of) {
    return false;
  } else if (current_seq_count > best_of) {
    LOG(FATAL) << "Request has more than " << best_of << " sequences.";
    return false;
  }

  CHECK(!sequences_.empty()) << "Request has no sequence.";
  const auto& seq = sequences_[0];
  // prefill is not finished, can not expand
  // FIXME later share_prefix
  if (!share_prefix ||
      seq->kv_state().kv_cache_tokens_num() >= seq->num_prompt_tokens()) {
    while (sequences_.size() < best_of) {
      add();
    }
    return true;
  }

  return false;
}

void SequencesGroup::generate_outputs(std::vector<SequenceOutput>& outputs,
                                      const Tokenizer& tokenizer,
                                      ThreadPool* thread_pool) {
  if (sequence_params_.streaming) {
    for (auto& seq : sequences_) {
      outputs.push_back(std::move(seq->generate_output()));
    }
    return;
  }

  auto n = sequence_params_.n;
  auto num = std::min(sequences_.size(), n);
  outputs.reserve(num);
  if (sequences_.size() > n && !check_beam_search()) {
    std::vector<std::pair<float, size_t>> logprobs_vec;
    logprobs_vec.reserve(sequences_.size());
    for (size_t i = 0; i < sequences_.size(); ++i) {
      logprobs_vec.emplace_back(sequences_[i]->get_average_logprob(), i);
    }
    std::sort(logprobs_vec.begin(),
              logprobs_vec.end(),
              [](const auto& l, const auto& r) { return l.first > r.first; });
    for (size_t i = 0; i < n; ++i) {
      const auto [logprob, index] = logprobs_vec[i];
      auto seq_output = sequences_[index]->generate_output(tokenizer);
      seq_output.index = i;
      outputs.push_back(std::move(seq_output));
    }
  } else {
    // speed up when using thread pool
    if (thread_pool && sequences_.size() > 1) {
      generate_outputs_parallel(outputs, tokenizer, thread_pool);
      return;
    }
    for (auto& seq : sequences_) {
      outputs.push_back(seq->generate_output(tokenizer));
    }
  }
}

void SequencesGroup::generate_outputs_parallel(
    std::vector<SequenceOutput>& outputs,
    const Tokenizer& tokenizer,
    ThreadPool* thread_pool) {
  size_t seq_size = sequences_.size();
  outputs.reserve(seq_size);
  size_t num_tasks = std::min(thread_pool->size(), seq_size);
  size_t num_tasks_pre =
      (seq_size + thread_pool->size() - 1) / thread_pool->size();
  std::vector<std::vector<SequenceOutput>> local_outputs;
  local_outputs.resize(num_tasks);

  auto gererate_output_local = [&](size_t work_id) {
    local_outputs[work_id].reserve(num_tasks_pre);
    for (size_t i = 0; i < num_tasks_pre; ++i) {
      size_t task_id = work_id * num_tasks_pre + i;
      if (task_id >= seq_size) break;

      const auto& base_seq = sequences_[task_id];
      local_outputs[work_id].emplace_back(base_seq->generate_output(tokenizer));
    }
  };

  BlockingCounter counter(num_tasks);

  {
    for (size_t i = 0; i < num_tasks; ++i) {
      thread_pool->schedule([gererate_output_local, i, &counter]() mutable {
        gererate_output_local(i);
        counter.decrement_count();
      });
    }

    counter.wait();
  }

  for (size_t i = 0; i < num_tasks; ++i) {
    outputs.insert(
        outputs.end(), local_outputs[i].begin(), local_outputs[i].end());
  }
}

void SequencesGroup::process_beam_search() {
  if (!check_beam_search()) {
    return;
  }
  Timer timer;
  size_t beam_width = sequence_params_.sampling_param->beam_width;
  size_t seq_size = sequences_.size();
  size_t topk = sequence_params_.sampling_param->top_logprobs;
  size_t num_candidates = topk * seq_size;

  if (num_candidates <= beam_width || seq_size == 1) {
    std::vector<std::unique_ptr<Sequence>> result;
    result.reserve(num_candidates);

    int32_t last_token_idx = sequences_[0]->num_tokens() - 1;
    size_t k = std::min(topk, beam_width);
    for (size_t i = 0; i < seq_size; i++) {
      std::unique_ptr<Sequence>& seq = sequences_[i];
      auto src_blocks = seq->kv_state().kv_blocks();
      const auto& top_logprobs =
          seq->logprob_state()->get_top_logprobs()[last_token_idx];
      const auto& top_tokens =
          seq->logprob_state()->get_top_tokens()[last_token_idx];
      for (int idx = 0; idx < k; ++idx) {
        result.emplace_back(std::make_unique<Sequence>(*(seq.get())));
        Token new_token(top_tokens[idx]);
        new_token.logprob = top_logprobs[idx];
        result.back()->update_token(last_token_idx, new_token);
        result.back()->kv_state().set_src_blocks(src_blocks,
                                                 /*need_swap*/ idx != 0);
      }
    }
    sequences_ = std::move(result);
    return;
  }

  SimpleTopKOptimizerBeamCandidate topk_optimizer(beam_width);
  int32_t last_token_idx = sequences_[0]->num_tokens() - 1;

  auto compute_candidates_for_sequence = [&](size_t i) {
    std::unique_ptr<Sequence>& seq = sequences_[i];
    int32_t num_generated_tokens = seq->num_generated_tokens();

    Slice<int32_t> token_ids = seq->tokens();
    const auto& log_probs = seq->logprob_state()->get_logprobs();
    const auto& top_logprobs =
        seq->logprob_state()->get_top_logprobs()[last_token_idx];
    const auto& top_tokens =
        seq->logprob_state()->get_top_tokens()[last_token_idx];
    float base_logprob =
        seq->get_average_logprob() * num_generated_tokens - top_logprobs[0];

    for (int idx = 0; idx < topk; ++idx) {
      float new_logprob = base_logprob + top_logprobs[idx];

      if (!topk_optimizer.worthInserting(new_logprob)) {
        break;
      }

      auto new_token_ids = std::vector<int32_t>(token_ids);
      new_token_ids[last_token_idx] = top_tokens[idx];
      auto new_log_probs = log_probs;
      new_log_probs[last_token_idx] = top_logprobs[idx];

      BeamCandidate candidate;
      candidate.seq_index = i;
      candidate.logprob_sum = new_logprob;
      candidate.token_ids = std::move(new_token_ids);
      candidate.logprobs = std::move(new_log_probs);
      topk_optimizer.insert(std::move(candidate));
    }
  };

  for (size_t i = 0; i < seq_size; ++i) {
    compute_candidates_for_sequence(i);
  }

  // std::vector<BeamCandidate> candidates = topk_optimizer.getTopKMove();
  std::vector<BeamCandidate> candidates = topk_optimizer.getTopKSorted();

  if (candidates.empty()) return;

  auto update_for_sequence = [&](size_t work_id, size_t num_tasks_pre) {
    std::unordered_set<int32_t> seq_idx_set;
    for (size_t i = 0; i < num_tasks_pre; ++i) {
      size_t task_id = work_id * num_tasks_pre + i;
      if (task_id >= beam_width) break;

      const BeamCandidate& c = candidates[task_id];
      auto& base_seq = sequences_[task_id];
      auto& src_seq = sequences_[c.seq_index];

      CHECK_EQ(base_seq->num_tokens(), c.token_ids.size());
      for (size_t token_idx = base_seq->num_prompt_tokens();
           token_idx < base_seq->num_tokens();
           token_idx++) {
        Token new_token(c.token_ids[token_idx]);
        new_token.logprob = c.logprobs[token_idx].has_value()
                                ? c.logprobs[token_idx].value()
                                : 0;
        base_seq->update_token(token_idx, new_token);
      }

      bool need_swap = false;
      if (seq_idx_set.find(c.seq_index) != seq_idx_set.end()) {
        need_swap = true;
      } else {
        seq_idx_set.insert(c.seq_index);
      }

      base_seq->logprob_state()->set_acc_logprob(c.logprob_sum);
      base_seq->logprob_state()->set_last_acc_token_idx(base_seq->num_tokens());

      auto src_blocks = src_seq->kv_state().kv_blocks();
      base_seq->kv_state().set_src_blocks(src_blocks, need_swap);
    }
  };

  CHECK_EQ(sequences_.size(), beam_width);
  update_for_sequence(0, beam_width);
  HISTOGRAM_OBSERVE(expand_beam_latency_microseconds,
                    timer.elapsed_microseconds());
}

void SequencesGroup::finish() {
  for (auto& sequence : sequences_) {
    sequence->finish();
  }
}

}  // namespace xllm
