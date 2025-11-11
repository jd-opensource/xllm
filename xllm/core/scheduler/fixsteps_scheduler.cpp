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

#include "fixsteps_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <folly/Unit.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "common/metrics.h"
#include "framework/batch/batch.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/engine.h"

namespace xllm {

namespace {
constexpr size_t kRequestQueueSize = 100000;
}  // namespace

FixStepsScheduler::FixStepsScheduler(Engine* engine, const Options& options)
    : ContinuousScheduler(engine, options) {}

bool FixStepsScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  if (request_queue_.write(request)) {  //.get()
    // take over the ownership of the request
    // request.release();
    return true;
  }
  // queue is full
  return false;
}

void FixStepsScheduler::handle_prefill_requests(
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    std::vector<std::shared_ptr<Request>>& finished_requests) {
  // Handle new request prompt first.
  // Include those requests that are preempted by others.
  //
  // schedule the prefill requests in the waiting priority queue until budgets
  // are exhausted.
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  //
  // NOTE: preempted requests will be pushed in waiting_priority_queue,
  // they may contian many sequences, so we should check here.
  bool budget_exhausted = false;
  bool blocks_exhausted = false;
  while (!waiting_priority_queue_.empty() && remaining_seq_budget > 0 &&
         remaining_token_budget > 0 &&
         kv_cache_manager_->kv_cache_utilization() <
             FLAGS_prefill_scheduling_memory_usage_threshold) {
    std::shared_ptr<Request> request(waiting_priority_queue_.top());
    if (request->finished() || request->cancelled()) {
      // kv_cache_manager_->deallocate(request.get());
      //  release the ownership of the request
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
    double allocated_estimate_latency = 0;
    bool can_schedule = true;
    std::vector<Sequence*> prefill_sequences;
    std::vector<size_t> prefill_sequences_budget;
    prefill_sequences.reserve(request->sequences().size());
    prefill_sequences_budget.reserve(request->sequences().size());
    for (auto& prefill_sequence : request->sequences()) {
      if (prefill_sequence->finished()) {
        continue;
      }

      size_t num_tokens = prefill_sequence->num_need_compute_tokens();
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        budget_exhausted = true;
        break;
      }

      prefill_sequences_budget.emplace_back(num_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += num_tokens;
      allocated_seqs += 1;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        // release shared blocks
        kv_cache_manager_->deallocate(seq);
      }
      break;
    }

    if (prefill_sequences.empty()) {
      continue;
    }

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
      running_queue_->empty()) {
    LOG(ERROR)
        << "Request prompt is too long, no enough budget/memory to schedule "
           "a single sequence.";
    // no enough memory to schedule single sequence, just finish the request
    std::shared_ptr<Request> request(waiting_priority_queue_.top());
    waiting_priority_queue_.pop();
    // block_manager_->release_blocks_for(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough budget to schedule single sequence."});
  }
}

std::vector<Batch> FixStepsScheduler::prepare_batch() {
  Timer timer;
  // propogate new requests to waiting_priority_queue_
  // Include those requests that are preempted by others.
  std::shared_ptr<Request> request;
  // read from request queue then push to waiting priority queue
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
      running_requests_.emplace_back(request);
    }
  }

  // handle finished/cancelled requests
  std::vector<std::shared_ptr<Request>> finished_requests;
  for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
       ++it) {
    if (*it == nullptr) {
      continue;
    }
    std::shared_ptr<Request> request = *it;
    request->update_connection_status();
    if (request->finished() || request->cancelled()) {
      // kv_cache_manager_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
    }
  }

  // clear previous batch
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  size_t num_preempted_requests = 0;

  handle_prefill_requests(
      remaining_token_budget, remaining_seq_budget, finished_requests);

  // only forward once, no decode requests
  // handle_decode_requests(
  //     remaining_token_budget, remaining_seq_budget, num_preempted_requests);

  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  // update the batch
  auto batches = BatchFactory::get_instance(options_.dp_size())
                     ->create_rec_batches(
                         running_requests_,
                         running_sequences_,
                         running_sequences_budgets_,
                         kv_cache_manager_->get_copy_in_cache_block_infos(),
                         kv_cache_manager_->get_copy_out_cache_block_infos(),
                         kv_cache_manager_->get_swap_cache_block_infos());

  // update metrics before returning
  if (!batches[0].empty()) {
    // only update the scheduling latency when there are requests to process
    COUNTER_ADD(scheduling_latency_seconds, timer.elapsed_seconds());
  }

  GAUGE_SET(num_pending_requests,
            pending_requests_.load(std::memory_order_relaxed));
  GAUGE_SET(num_running_requests, running_requests_.size());
  GAUGE_SET(num_waiting_requests,
            waiting_priority_queue_.size() + running_queue_->size());

  GAUGE_ADD(num_preempted_requests, num_preempted_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_->kv_cache_utilization());
  if (!FLAGS_enable_continuous_kvcache) {
    GAUGE_SET(num_blocks_in_prefix_cache,
              kv_cache_manager_->num_blocks_in_prefix_cache().size());
    GAUGE_SET(num_free_blocks, kv_cache_manager_->num_free_blocks().size());
    GAUGE_SET(num_used_blocks, kv_cache_manager_->num_used_blocks().size());
  }
  return batches;
}

std::vector<Batch> FixStepsScheduler::schedule_request(
    const absl::Duration& timeout) {
  const auto deadline = absl::Now() + timeout;
  std::vector<Batch> batch;
  while (true) {
    batch = prepare_batch();
    bool all_empty =
        std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
          return one_batch.empty();
        });
    if (!all_empty) {
      return batch;
    }
    const auto now = absl::Now();
    if (now > deadline) {
      break;
    }
    // wait for new requests to arrive
    constexpr uint64_t kStepSleepTimeMs = 1;
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }
  // return an empty batch
  return batch;
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void FixStepsScheduler::step(const absl::Duration& timeout) {
  if (!options_.enable_schedule_overlap()) {
    // get a new batch of requests
    std::vector<Batch> batch = schedule_request(timeout);
    bool all_empty =
        std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
          return one_batch.empty();
        });
    if (all_empty) {
      return;
    }
    engine_->step(batch);
    kv_cache_manager_->reset_copy_content();
  } else {
    LOG(ERROR) << "FixStepsScheduler::step() not supported with "
                  "enable_schedule_overlap";
  }
}

}  // namespace xllm