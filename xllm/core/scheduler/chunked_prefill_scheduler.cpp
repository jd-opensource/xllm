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

#include "scheduler/chunked_prefill_scheduler.h"

#include "common/metrics.h"
#include "framework/batch/batch_factory.h"
#include "util/timer.h"
#include "util/utils.h"

DEFINE_int32(chunked_match_frequency,
             2,
             "Number of sequence prefix cache match frequency");
namespace xllm {

ChunkedPrefillScheduler::ChunkedPrefillScheduler(Engine* engine,
                                                 const Options& options)
    : ContinuousScheduler(engine, options) {}

ChunkedPrefillScheduler::~ChunkedPrefillScheduler() {
  // release all requests in the priority queue
  while (!waiting_priority_queue_.empty()) {
    waiting_priority_queue_.pop();
  }

  // release all requests in the running priority queue
  while (!running_queue_.empty()) {
    running_queue_.pop_front();
  }
}

// NOTE: refactor ChunkedPrefillScheduler and ContinuousScheduler later.
void ChunkedPrefillScheduler::handle_abnormal_request(
    const std::vector<Sequence*>& candidate_sequences,
    const std::vector<size_t>& candidate_token_budgets,
    const size_t& allocated_tokens,
    const size_t& allocated_seqs,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    bool budget_exhausted,
    bool blocks_exhausted) {
  std::shared_ptr<Request> request = running_queue_.front();
  if (candidate_sequences.empty()) {
    if (!running_sequences_.empty()) {
      return;
    }

    // unknown case, maybe a schdule bug.
    if (budget_exhausted && blocks_exhausted) {
      LOG(FATAL) << "Unknown case, budget and blocks are not exhausted, but "
                    "there are no running sequences."
                 << " budget_exhausted = " << budget_exhausted
                 << " blocks_exhausted = " << blocks_exhausted
                 << " candidate_sequences.size = " << candidate_sequences.size()
                 << ", running_sequences.size = " << running_sequences_.size();
    }

    // budget exhausted
    if (budget_exhausted) {
      LOG(ERROR) << "Request prompt is too long, please set a larger "
                    "max_tokens value via --max_tokens_per_batch.";
    } else {
      CHECK(running_queue_.size() == 1)
          << "Running queue size is not 1, there maybe a bug of request "
             "preemption logic. running_queue_.size ="
          << running_queue_.size();
      if (util::sum(block_manager_->num_used_blocks()) !=
          request->total_num_blocks()) {
        // blocks_exhausted is true.
        // NOTE: consider dp > 1, here we need get all num blocks in use.
        // Total num blocks in use not equal request->total_num_blocks() means
        // some sequences are not scheduled but hold blocks in disagg PD mode.
        return;
      }
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                 << "a single sequence.";
    }

    // request is too long, budget or memory no enough.
    running_queue_.pop_front();
    block_manager_->deallocate(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough resource to schedule a single sequence"});
  } else {
    // partially schedule the sequences in request
    running_queue_.pop_front();
    running_requests_.emplace_back(request);
    running_sequences_.insert(running_sequences_.end(),
                              candidate_sequences.begin(),
                              candidate_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      candidate_token_budgets.begin(),
                                      candidate_token_budgets.end());
    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
  }
}

void ChunkedPrefillScheduler::handle_running_queue_requests(
    const size_t max_tokens_per_chunk_for_prefill,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_preempted_requests,
    std::vector<Sequence*>& prefill_stage_sequences,
    bool& budget_exhausted,
    bool& blocks_exhausted) {
  while (!running_queue_.empty() &&
         remaining_token_budget > options_.num_speculative_tokens() &&
         remaining_seq_budget > 0) {
    std::shared_ptr<Request> request(running_queue_.front());
    // TODO: check if request is timeout

    const size_t num_sequences = request->sequences().size();
    std::vector<Sequence*> candidate_sequences;
    std::vector<size_t> candidate_token_budgets;
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    bool has_enough_budget = true;
    bool has_enough_blocks = true;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;

    for (auto& sequence : request->sequences()) {
      // skip finished sequence.
      if (sequence->finished()) {
        continue;
      }

      // no budget left
      if (allocated_seqs >= remaining_seq_budget ||
          allocated_tokens + options_.num_speculative_tokens() >=
              remaining_token_budget) {
        has_enough_budget = false;
        break;
      }

      // The max tokens current sequence can handle.
      const size_t assume_max_tokens =
          std::min(max_tokens_per_chunk_for_prefill,
                   remaining_token_budget - allocated_tokens);
      // actual tokens be handled,
      // decode: 1 + num_speculative_tokens
      // prefill: std::min(seq.num_tokens(), min_tokens_per_iter_for_prefill)
      size_t current_step_handle_tokens = 0;
      // no budget left
      if (!allocate_blocks_for(
              sequence.get(), assume_max_tokens, &current_step_handle_tokens)) {
        has_enough_blocks = false;
        break;
      }

      // if (sequence->if_cache_block_for_prefill()) {
      //   block_manager_->cache(sequence.get());
      // }

      // the new request do chunked prefill
      if (sequence->kv_state().kv_cache_tokens_num() == 0 ||
          sequence->is_prefill_stage()) {
        prefill_stage_sequences.emplace_back(sequence.get());
      }

      // update the allocated tokens for the sequence
      allocated_tokens += current_step_handle_tokens;
      allocated_seqs += 1;
      candidate_sequences.emplace_back(sequence.get());
      candidate_token_budgets.emplace_back(current_step_handle_tokens);
    }
    CHECK(allocated_tokens <= remaining_token_budget);
    CHECK(allocated_seqs <= remaining_seq_budget);

    if (has_enough_budget && has_enough_blocks) {
      // remove the request from the priority queue
      running_queue_.pop_front();
      // add the request to the batch
      running_requests_.emplace_back(request);
      running_sequences_.insert(running_sequences_.end(),
                                candidate_sequences.begin(),
                                candidate_sequences.end());
      running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                        candidate_token_budgets.begin(),
                                        candidate_token_budgets.end());
      remaining_token_budget -= allocated_tokens;
      remaining_seq_budget -= allocated_seqs;
      continue;
    }

    // budget exhausted, do partially schedule the request
    if (!has_enough_budget) {
      handle_abnormal_request(candidate_sequences,
                              candidate_token_budgets,
                              allocated_tokens,
                              allocated_seqs,
                              remaining_token_budget,
                              remaining_seq_budget,
                              true, /*budget_exhausted*/
                              false /*blocks_exhausted*/);
      budget_exhausted = true;
      break;
    }

    // memory exhausted, preempt lowest priority request and retry.
    // preemptable_requests_ only contain decoding requests.
    if (running_queue_.size() > 1) {
      std::shared_ptr<Request> request_to_preempt = running_queue_.back();

      if (request_to_preempt.get() != request.get()) {
        ++num_preempted_requests;
        block_manager_->deallocate(request_to_preempt.get());
        running_queue_.pop_back();
        // add preemptable request to waiting priority queue
        request_to_preempt->set_preempted();
        waiting_priority_queue_.push(request_to_preempt);
      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }

      continue;
    }

    // no requests left to preempt
    handle_abnormal_request(candidate_sequences,
                            candidate_token_budgets,
                            allocated_tokens,
                            allocated_seqs,
                            remaining_token_budget,
                            remaining_seq_budget,
                            false, /*budget_exhausted*/
                            true /*blocks_exhausted*/);
    blocks_exhausted = true;
    break;
  }
}

void ChunkedPrefillScheduler::handle_prefill_requests(
    const size_t max_tokens_per_chunk_for_prefill,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    std::vector<Sequence*>& prefill_stage_sequences,
    bool& blocks_exhausted,
    std::vector<std::shared_ptr<Request>>& finished_requests) {
  // NOTE: preempted requests will be pushed in waiting_priority_queue,
  // they may contian many sequences, so we should check here.
  while (!waiting_priority_queue_.empty() && remaining_token_budget > 0 &&
         remaining_seq_budget > 0) {
    std::shared_ptr<Request> request(waiting_priority_queue_.top());
    if (request->finished() || request->cancelled()) {
      block_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // remove the request from the priority queue
      waiting_priority_queue_.pop();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1)
          << "Waiting request should have only one sequence.";
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      size_t num_tokens = prefill_sequence->num_tokens();
      // The max tokens current sequence can handle.
      const size_t assume_max_tokens =
          std::min(max_tokens_per_chunk_for_prefill,
                   remaining_token_budget - allocated_tokens);
      num_tokens = std::min(assume_max_tokens, num_tokens);
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        break;
      }
      size_t current_step_handle_tokens = 0;
      if (!allocate_blocks_for(prefill_sequence.get(),
                               num_tokens,
                               &current_step_handle_tokens)) {
        // release shared blocks
        block_manager_->deallocate(prefill_sequence.get());
        can_schedule = false;
        blocks_exhausted = true;
        break;
      }
      prefill_sequences_budget.emplace_back(current_step_handle_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += current_step_handle_tokens;
      allocated_seqs += 1;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        // release shared blocks
        block_manager_->deallocate(seq);
      }
      break;
    }

    if (prefill_sequences.empty()) {
      continue;
    }

    prefill_stage_sequences.insert(prefill_stage_sequences.end(),
                                   prefill_sequences.begin(),
                                   prefill_sequences.end());
    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    waiting_priority_queue_.pop();
    running_requests_.emplace_back(request);
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
  }

  if (running_sequences_.empty() && !waiting_priority_queue_.empty() &&
      running_queue_.empty() && block_manager_->kv_cache_utilization() == 0) {
    LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                  "a single sequence";
    // no enough memory to schedule single sequence, just finish the request
    std::shared_ptr<Request> request(waiting_priority_queue_.top());
    waiting_priority_queue_.pop();
    block_manager_->deallocate(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough memory to schedule single sequence"});
  }
}

void ChunkedPrefillScheduler::handle_remaining_budget(
    size_t& remaining_token_budget,
    std::vector<Sequence*>& prefill_stage_sequences,
    bool& blocks_exhausted) {
  size_t prefill_stage_seq_idx = 0;
  for (size_t i = 0; i < running_sequences_.size() &&
                     prefill_stage_seq_idx < prefill_stage_sequences.size();
       ++i) {
    if (prefill_stage_sequences[prefill_stage_seq_idx] !=
        running_sequences_[i]) {
      continue;
    }
    ++prefill_stage_seq_idx;
    Sequence* sequence = running_sequences_[i];
    size_t& token_budget = running_sequences_budgets_[i];

    // add previous allocated tokens back
    remaining_token_budget += token_budget;
    size_t current_step_handle_tokens = 0;
    // no memory left
    if (!allocate_blocks_for(
            sequence, remaining_token_budget, &current_step_handle_tokens)) {
      blocks_exhausted = true;
      break;
    }
    // update the allocated tokens for the sequence
    token_budget = current_step_handle_tokens;
    CHECK(remaining_token_budget >= current_step_handle_tokens);
    remaining_token_budget -= current_step_handle_tokens;

    // no budget left
    if (remaining_token_budget == 0) {
      break;
    }
  }
}

std::vector<Batch> ChunkedPrefillScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to waiting_priority_queue_
  std::shared_ptr<Request> request;
  // read from request queue then push to waiting_priority_queue_
  while (request_queue_.read(request)) {
    CHECK(request);

    // expand sequences to the target number if prefix cache is disabled.
    if (!enable_prefix_cache_) {
      // expand sequences to the target number
      request->expand_sequences(false);
    }

    if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
      waiting_priority_queue_.push(request);
    } else {
      // request from prefill instance in disagge pd mode.
      // NOTE: running_requests_ keep a batch of requests in running state,
      //   sorted by priority from high to low.
      running_requests_.emplace_back(request);
    }
  }

  // handle finished/cancelled requests
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    std::shared_ptr<Request> request = *it;
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      block_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
    }
  }

  // insert running requests back to the priority queue, iterating from the
  // lowest priority to the highest
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    // finished request is set to nullptr
    if (*it == nullptr) {
      continue;
    }
    std::shared_ptr<Request> request = *it;
    if (request->finished() || request->cancelled()) {
      LOG(FATAL) << "Unknow error, finished/cancelled request have be handled "
                    "before. request_id is "
                 << request->request_id();
    }

    // check if the request can be expanded
    if (request->expand_sequences()) {
      // cache the blocks to share among the sequences
      block_manager_->cache(request->sequences()[0].get());
    }

    // release blocks for finished sequences here
    for (auto& sequence : request->sequences()) {
      if (sequence->finished()) {
        block_manager_->deallocate(sequence.get());
      }
    }

    // push the request front to the priority deque
    running_queue_.push_front(request);
  }

  // clear previous batch
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  // for new request which in prefill stage, this means we handle
  // std::min(seq.num_tokens(), max_tokens_per_chunk_for_prefill)
  // num tokens per step.
  // seq.num_tokens(): `prompt tokens` or `(prompt tokens + partial
  // generated tokens when request be preempted and recomputed)`.
  //
  // If we don't set `max_tokens_per_chunk_for_prefill`, the default value is
  // int32_t::max(), which means we process high-priority prefill first.
  // If set `max_tokens_per_chunk_for_prefill`, for example 2048,
  // If a high-priority prefill request is very long, each step only
  // handle 2028 promote tolkens, the scheduler will allocate
  // part of the resources to subsequent prefill requests
  // for execution, thereby preventing subsequent requests from being
  // blocked by the first extremely long request.
  //
  // NOTE: MagicNum 64 here, avoid users setting small value.
  const size_t max_tokens_per_chunk_for_prefill =
      std::max(options_.max_tokens_per_chunk_for_prefill(), 64);

  // Max tokens be handled in once chunked prefill schedule.
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = options_.max_seqs_per_batch();
  size_t num_preempted_requests = 0;
  bool budget_exhausted = false;
  bool blocks_exhausted = false;
  // keep the requests in prefill stage
  std::vector<Sequence*> prefill_stage_sequences;

  // step-1. handle requests from running_queue_
  // "decode-maximal batching" principle:
  //  Prioritize adding Decode requests to the batch until
  //  the KV cache space they occupy reaches the upper limit.
  //  Subsequently, based on the remaining token quota,
  //  split sequences requiring Prefill into chunks and
  //  add them to the batch.
  //
  handle_running_queue_requests(max_tokens_per_chunk_for_prefill,
                                remaining_token_budget,
                                remaining_seq_budget,
                                num_preempted_requests,
                                prefill_stage_sequences,
                                budget_exhausted,
                                blocks_exhausted);

  // step-2. handle new prefill request
  // new prefill request can not preempt any running requests.
  //
  if (!budget_exhausted && !blocks_exhausted) {
    handle_prefill_requests(max_tokens_per_chunk_for_prefill,
                            remaining_token_budget,
                            remaining_seq_budget,
                            prefill_stage_sequences,
                            blocks_exhausted,
                            finished_requests);
  }

  // step-3. remaining_token_budget > 0, try to allocate more tokens to
  // prefill stage reuquests.
  if (remaining_token_budget > 0 && !blocks_exhausted) {
    handle_remaining_budget(
        remaining_token_budget, prefill_stage_sequences, blocks_exhausted);
  }

  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(running_sequences_, running_sequences_budgets_);

  if (!batches[0].empty()) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests,
            waiting_priority_queue_.size() + running_queue_.size());
  GAUGE_SET(num_preempted_requests, num_preempted_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc, block_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(block_manager_->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks, util::max(block_manager_->num_free_blocks()));
  GAUGE_SET(num_used_blocks, util::min(block_manager_->num_used_blocks()));

  return batches;
}

bool ChunkedPrefillScheduler::allocate_blocks_for(
    Sequence* sequence,
    size_t token_budget,
    size_t* current_step_handle_tokens) {
  // token budget should be large enough for one speculative decoding step
  CHECK_GT(token_budget, options_.num_speculative_tokens());

  if (sequence->kv_state().num_kv_blocks() == 0) {
    // allocate shared blocks
    block_manager_->allocate_shared(sequence);
  }
  allocate_shared_blocks_for(sequence);

  // number of tokens in the kv cache, which are already processed
  const size_t kv_cache_tokens_num = sequence->kv_state().kv_cache_tokens_num();
  // the total number tokens for the sequence can be handled till now.
  // there may some tokens can not be handled once when enable chunked prefill.
  size_t max_handle_num_tokens =
      std::min(kv_cache_tokens_num + token_budget, sequence->num_tokens());

  // speculative decoding specific logic,
  // prefill stage don't need speculative decoding.
  //
  // if in decoding stage
  if (options_.num_speculative_tokens() > 0 && !sequence->is_prefill_stage() &&
      kv_cache_tokens_num > 0) {
    max_handle_num_tokens += options_.num_speculative_tokens();
  }

  // make sure the sequence proceeds forward
  CHECK_GT(max_handle_num_tokens, kv_cache_tokens_num);

  // the actual allocated tokens is the difference between the total
  // number of tokens and the number of tokens already processed
  *current_step_handle_tokens = max_handle_num_tokens - kv_cache_tokens_num;
  // allocate blocks for the sequence
  return block_manager_->allocate(sequence, max_handle_num_tokens);
}

void ChunkedPrefillScheduler::allocate_shared_blocks_for(Sequence* sequence) {
  if (sequence->kv_state().num_kv_blocks() == 0) {
    // allocate shared blocks
    block_manager_->allocate_shared(sequence);
    return;
  }
  if (sequence->is_prefill_stage()) {
    const size_t max_tokens_per_chunk_for_prefill =
        std::max(options_.max_tokens_per_chunk_for_prefill(), 64);
    size_t total_chunked_size =
        (sequence->num_tokens() + max_tokens_per_chunk_for_prefill - 1) /
        max_tokens_per_chunk_for_prefill;
    if (total_chunked_size < FLAGS_chunked_match_frequency) {
      block_manager_->allocate_shared(sequence);
      return;
    }
    size_t prefix_cache_interval =
        (total_chunked_size + FLAGS_chunked_match_frequency - 1) /
        FLAGS_chunked_match_frequency;
    size_t cur_chunked_index = sequence->kv_state().kv_cache_tokens_num() /
                               max_tokens_per_chunk_for_prefill;
    if (cur_chunked_index % prefix_cache_interval == 0) {
      // allocate shared blocks
      block_manager_->allocate_shared(sequence);
    }
  }
}

}  // namespace xllm
