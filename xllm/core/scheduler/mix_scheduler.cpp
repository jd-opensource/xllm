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

#include "scheduler/mix_scheduler.h"

#include <algorithm>
#include <limits>

#include "common/metrics.h"
#include "framework/batch/batch_factory.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

MixScheduler::MixScheduler(Engine* engine, const Options& options)
    : ChunkedPrefillScheduler(engine, options) {}

MixScheduler::~MixScheduler() {
  // release all requests in the running priority queue
  while (!running_queue_.empty()) {
    running_queue_.pop_front();
  }
}

//----------------------------------

void MixScheduler::get_latency_budget_and_request_order(
    std::list<std::shared_ptr<Request>>& running_queue,
    double& latency_budget) {
  // update request metrcis
  for (auto& request : running_queue) {
    auto& sequence = request->sequences()[0];
    sequence->set_estimated_latency(
        profile_manager_->predict_step_time(sequence.get(), false));
    request->set_elapsed_time_ms();
    request->set_deadline_ms();
  }

  auto constant_overhead = profile_manager_->get_constant_overhead();
  double total_exec_time = 0.0;

  int32_t min_remaining_time = std::numeric_limits<int32_t>::max();
  int32_t min_tpot = std::numeric_limits<int32_t>::max();
  for (auto it = running_queue.begin(); it != running_queue.end(); it++) {
    auto request = *it;
    auto& sequence = request->sequences()[0];
    auto remaining_time = request->get_remaining_time();
    total_exec_time += sequence->estimated_latency();
    if (request->tpot_slo_ms() < min_tpot) {
      min_tpot = static_cast<int32_t>(request->tpot_slo_ms());
    }

    if (remaining_time < sequence->estimated_latency() + constant_overhead) {
      // Currently, overdue and urgent requests are being handled together.
      continue;
    }
    if (remaining_time < min_remaining_time) {
      min_remaining_time = static_cast<int32_t>(remaining_time);
    }
  }
  // determine latency budget
  // int32_t threshold = static_cast<int32_t>(4 * constant_overhead);
  int32_t threshold = static_cast<int32_t>(0.65 * min_tpot);
  latency_budget = std::max(min_remaining_time, threshold);

  for (auto& request : running_queue) {  // determine urgency
    auto& sequence = request->sequences()[0];
    if (request->get_remaining_time() <
        total_exec_time * latency_budget /
            (latency_budget - constant_overhead)) {
      request->set_urgency(Urgency::URGENT);
    } else {
      request->set_urgency(Urgency::NORMAL);
    }
  }

  // sort, should be urgency_density now
  running_queue.sort(create_comparator(options_.priority_strategy(), true));
}

void MixScheduler::handle_running_queue_requests(
    double& latency_budget,
    double& estimate_latency,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_preempted_requests,
    std::vector<Sequence*>& prefill_stage_sequences,
    std::list<shared_ptr<Request>>& running_queue,
    bool& budget_exhausted,
    bool& blocks_exhausted) {
  if (running_queue.empty()) {
    return;
  }

  get_latency_budget_and_request_order(running_queue, latency_budget);

  std::vector<std::shared_ptr<Request>> preemeted_request_vec;
  while (!running_queue.empty() &&
         remaining_token_budget > options_.num_speculative_tokens() &&
         latency_budget > estimate_latency && remaining_seq_budget > 0) {
    std::shared_ptr<Request> request(running_queue.front());

    const size_t num_sequences = request->sequences().size();
    CHECK(num_sequences == 1) << "currently only support one sequence.";

    std::vector<Sequence*> candidate_sequences;
    std::vector<size_t> candidate_token_budgets;
    candidate_sequences.reserve(num_sequences);
    candidate_token_budgets.reserve(num_sequences);

    auto constant_overhead = profile_manager_->get_constant_overhead();

    budget_exhausted = false;
    blocks_exhausted = false;
    size_t allocated_tokens = 0;
    size_t allocated_seqs = 0;
    double allocated_estimate_latency = 0;
    for (auto& sequence : request->sequences()) {
      // skip finished sequence.
      if (sequence->finished()) {
        continue;
      }
      CHECK(!sequence->finished());

      size_t assume_max_tokens = remaining_token_budget - allocated_tokens;
      size_t num_tokens = sequence->num_tokens();
      size_t kv_cache_tokens_num = sequence->kv_state().kv_cache_tokens_num();

      if (options_.enable_latency_aware_schedule()) {
        if (sequence->is_prefill_stage()) {
          assume_max_tokens = get_max_chunk(
              sequence.get(),
              static_cast<int32_t>(latency_budget - estimate_latency));
          if (assume_max_tokens == kv_cache_tokens_num) {
            budget_exhausted = true;
            break;
          }
          if (assume_max_tokens != num_tokens &&
              (assume_max_tokens - kv_cache_tokens_num) <= 50) {
            // Preventing the creation of too many small chunks (this may not be
            // necessary).
            budget_exhausted = true;
            break;
          }
          allocated_estimate_latency = profile_manager_->predict_step_time(
              assume_max_tokens, kv_cache_tokens_num, false);
        } else {
          allocated_estimate_latency =
              profile_manager_->predict_step_time(sequence.get(), false);
          if (estimate_latency + allocated_estimate_latency > latency_budget) {
            // slack is too small, even decode is unable to include
            budget_exhausted = true;
            break;
          }
        }
      }

      if (allocated_seqs + 1 > remaining_seq_budget ||
          allocated_tokens + assume_max_tokens > remaining_token_budget) {
        budget_exhausted = true;
        break;
      }

      size_t current_step_handle_tokens = 0;
      // no budget left
      if (!allocate_blocks_for(
              sequence.get(), assume_max_tokens, &current_step_handle_tokens)) {
        blocks_exhausted = true;
        break;
      }

      // update the allocated tokens for the sequence
      allocated_tokens += current_step_handle_tokens;
      allocated_seqs += 1;
      candidate_sequences.emplace_back(sequence.get());
      candidate_token_budgets.emplace_back(current_step_handle_tokens);
    }
    if (!blocks_exhausted && !budget_exhausted) {
      // remove the request from the priority queue
      // running_queue.pop_top();
      running_queue.pop_front();
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
      estimate_latency += allocated_estimate_latency;
      continue;
    }

    if (budget_exhausted) {
      if (candidate_sequences.empty() && running_sequences_.empty()) {
        LOG(ERROR) << "Request prompt = "
                   << request->sequences()[0]->num_tokens()
                   << " is too long, please set a larger "
                      "max_tokens value via --max_tokens_per_batch.";
        running_queue.pop_front();
        kv_cache_manager_->deallocate(request.get());
        response_processor_->process_failed_request(
            request,
            {StatusCode::RESOURCE_EXHAUSTED,
             "No enough resource to schedule a single sequence"});
      }
      break;
    }

    // memory exhausted, preempt lowest priority request and retry.
    // preemptable_requests_ only contain decoding requests.
    // Maybe improve: for prefill stage sequence, we know how many blocks it
    // needs
    bool find_preempt = false;
    auto running_queue_iterator = std::prev(running_queue.end());
    while (running_queue_iterator != running_queue.begin()) {
      std::shared_ptr<Request> request_to_preempt = *running_queue_iterator;
      if (request_to_preempt.get() != request.get()) {
        if (request_to_preempt->sequences()[0]
                ->kv_state()
                .kv_cache_tokens_num() != 0) {
          // If KV cache is available, preempt.
          ++num_preempted_requests;
          // TO IMPROVE: kv cache offload to cpu
          kv_cache_manager_->deallocate(request_to_preempt.get());
          auto prev_running_queue_iterator = running_queue_iterator;
          running_queue_iterator--;
          running_queue.erase(prev_running_queue_iterator);
          // add preemptable request to waiting priority queue
          request_to_preempt->set_preempted();
          preemeted_request_vec.push_back(request_to_preempt);
          find_preempt = true;
          break;
        } else {
          // indicates it is prefill, skip
          running_queue_iterator--;
        }

      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }
    }
    if (find_preempt) {
      continue;
    }
    // no enough memeory to preempt
    if (candidate_sequences.empty() && running_sequences_.empty()) {
      LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                 << "a single sequence.";
      running_queue.pop_front();
      kv_cache_manager_->deallocate(request.get());
      response_processor_->process_failed_request(
          request,
          {StatusCode::RESOURCE_EXHAUSTED,
           "No enough resource to schedule a single sequence"});
    }
    blocks_exhausted = true;
    break;
  }

  // push back to running queue
  while (!preemeted_request_vec.empty()) {
    std::shared_ptr<Request> request(preemeted_request_vec.back());
    running_queue.push_back(request);
    preemeted_request_vec.pop_back();
  }
}

std::vector<Batch> MixScheduler::prepare_batch() {
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
      // both prefill and decode in one queue
      running_queue_.push_back(request);
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
      kv_cache_manager_->deallocate(request.get());
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
    handle_running_requests(*it);
    // unified multi priority strategy
    running_queue_.push_back(*it);
  }

  // clear previous batch
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  double latency_budget = options_.max_global_tpot_ms();

  // constant overhead ahead
  double estimate_latency = profile_manager_->get_constant_overhead();
  // Max tokens be handled in once chunked prefill schedule.
  size_t remaining_token_budget = options_.enable_profile_token_budget()
                                      ? profile_manager_->get_token_budget()
                                      : options_.max_tokens_per_batch();
  size_t remaining_seq_budget = options_.max_seqs_per_batch();
  size_t num_preempted_requests = 0;
  bool budget_exhausted = false;
  bool blocks_exhausted = false;
  // keep the requests in prefill stage
  std::vector<Sequence*> prefill_stage_sequences;

  handle_running_queue_requests(latency_budget,
                                estimate_latency,
                                remaining_token_budget,
                                remaining_seq_budget,
                                num_preempted_requests,
                                prefill_stage_sequences,
                                running_queue_,
                                budget_exhausted,
                                blocks_exhausted);

  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_batches(running_requests_,
                                      running_sequences_,
                                      running_sequences_budgets_);

  if (!batches[0].empty()) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  if (num_preempted_requests > 0) {
    LOG(INFO) << "Number of preempted requests in this round: "
              << num_preempted_requests;
  }
  GAUGE_SET(num_preempted_requests, num_preempted_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_->kv_cache_utilization());
  GAUGE_SET(num_blocks_in_prefix_cache,
            util::min(kv_cache_manager_->num_blocks_in_prefix_cache()));
  GAUGE_SET(num_free_blocks, util::max(kv_cache_manager_->num_free_blocks()));
  GAUGE_SET(num_used_blocks, util::min(kv_cache_manager_->num_used_blocks()));

  return batches;
}

int32_t MixScheduler::get_max_chunk(Sequence* sequence,
                                    int32_t latency_budget,
                                    bool use_quadratic_formula) {
  size_t num_tokens = sequence->num_tokens();
  size_t kv_cache_tokens_num = sequence->kv_state().kv_cache_tokens_num();
  if (profile_manager_->predict_step_time(
          num_tokens, kv_cache_tokens_num, false) <= latency_budget) {
    return num_tokens;
  }
  if (latency_budget <= 0) {
    return kv_cache_tokens_num;
  }
  if (use_quadratic_formula) {
    // use quadratic formula to get root
    return profile_manager_->get_quadratic_root(sequence, latency_budget);
  } else {
    // use binary search

    int32_t left = kv_cache_tokens_num + 1;
    int32_t right = num_tokens + 1;
    // [left, right)
    while (left < right) {
      int32_t mid = left + (right - left) / 2;
      auto predict_time =
          profile_manager_->predict_step_time(mid, kv_cache_tokens_num, false);
      if (predict_time <= latency_budget) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left - 1;
  }
}

bool MixScheduler::if_queue_not_empty() { return !running_queue_.empty(); }

}  // namespace xllm
