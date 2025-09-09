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

#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <folly/MPMCQueue.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "common/metrics.h"
#include "framework/batch/batch_factory.h"
#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/engine.h"
#include "scheduler/decode_priority_queue.h"
#include "util/utils.h"

namespace xllm {
namespace {
constexpr size_t kRequestQueueSize = 100000;
}  // namespace

ContinuousScheduler::ContinuousScheduler(Engine* engine, const Options& options)
    : options_(options),
      engine_(engine),
      request_queue_(kRequestQueueSize),
      waiting_priority_queue_(create_comparator(options.priority_strategy())),
      waiting_priority_queue_offline_(
          create_comparator(options.priority_strategy())) {
  CHECK(engine_ != nullptr);

  if (!FLAGS_enable_continuous_kvcache) {
    auto block_manager_pool = engine_->block_manager_pool();
    CHECK(block_manager_pool != nullptr);
    kv_cache_manager_client_ =
        std::make_unique<KVCacheManagerClient>(block_manager_pool);
  } else {
    auto page_manager_pool = engine_->page_manager_pool();
    CHECK(page_manager_pool != nullptr);
    kv_cache_manager_client_ =
        std::make_unique<KVCacheManagerClient>(page_manager_pool);
  }

  enable_prefix_cache_ = FLAGS_enable_prefix_cache;

  last_batch_.resize(options_.dp_size());

  response_processor_ = std::make_unique<AsyncResponseProcessor>(
      engine_->tokenizer(),
      options_.instance_role(),
      options_.enable_schedule_overlap(),
      options_.enable_decode_response_to_service());
  create_running_queue(options);
  if (options_.enable_service_routing()) {
    XServiceClient::get_instance()->set_scheduler(this);
  }

  instance_info_.name = options_.instance_name().value_or("");
  instance_info_.type = options_.instance_role().value().to_string();
  instance_info_.dp_size = options.dp_size();
}

ContinuousScheduler::~ContinuousScheduler() { running_requests_.clear(); }

bool ContinuousScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  prepare_cache_async(request);

  if (request_queue_.write(request)) {
    return true;
  }

  return false;
}

void ContinuousScheduler::create_running_queue(const Options& options) {
  if (options.priority_strategy() == "FCFS") {
    running_queue_offline_ = std::make_unique<FCFSQueue>();
    running_queue_ = std::make_unique<FCFSQueue>();
  } else {
    std::unique_ptr<PriorityComparator> comparator;
    if (options.priority_strategy() == "deadline") {
      comparator = std::make_unique<DeadlineComparator>();
    } else if (options.priority_strategy() == "priority") {
      comparator = std::make_unique<StrictPriorityComparator>();
    } else {
      LOG(FATAL) << "Unknown strategy: " << options.priority_strategy();
    }
    running_queue_ =
        std::make_unique<DynamicPriorityQueue>(std::move(comparator));
    running_queue_offline_ =
        std::make_unique<DynamicPriorityQueue>(std::move(comparator));
  }
}

bool ContinuousScheduler::check_if_enough_to_evict(
    DecodePriorityQueue* running_queue_to_evict,
    Sequence* prefill_sequence,
    size_t max_handle_num_tokens,
    size_t& num_request_to_evict) {
  // check if it's enough when we evict this requests queue
  // TODO(zhangxu)
  auto block_size = kv_cache_manager_client_->options().block_size();
  const size_t num_blocks_needed =
      (max_handle_num_tokens + block_size - 1) / block_size;
  size_t num_blocks_can_evict = 0;
  // count the number of blocks can be preempted
  for (auto it = running_queue_to_evict->rbegin();
       it != running_queue_to_evict->rend();
       ++it) {
    std::shared_ptr<Request> request_to_preempt = *it;
    num_request_to_evict++;
    // count the number of blocks belong to the request
    for (const auto& seq : request_to_preempt->sequences()) {
      // num_blocks_can_evict += seq->kv_state().num_kv_blocks();
      size_t shared_kv_blocks_num = seq->kv_state().shared_kv_blocks_num();
      size_t num_kv_blocks = seq->kv_state().num_kv_blocks();
      CHECK_GE(num_kv_blocks, shared_kv_blocks_num);
      for (size_t i = 0; i < shared_kv_blocks_num; i++) {
        // if ==2, prefix cache block will be evicted when allocate
        const auto& block = seq->kv_state().kv_blocks()[i];
        if (block.ref_count() <= 2) {
          num_blocks_can_evict += 1;
        }
      }
      num_blocks_can_evict += (num_kv_blocks - shared_kv_blocks_num);
    }
    if (num_blocks_needed <= num_blocks_can_evict) {
      return true;
    }
  }
  return false;
}

void ContinuousScheduler::handle_prefill_requests(
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    RequestPriorityQueue& waiting_priority_queue,
    size_t& num_online_prefill_preempt_offline_requests,
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
  while (!waiting_priority_queue.empty() && remaining_seq_budget > 0 &&
         remaining_token_budget > 0 &&
         kv_cache_manager_client_->kv_cache_utilization() <
             FLAGS_prefill_scheduling_memory_usage_threshold) {
    std::shared_ptr<Request> request(waiting_priority_queue.top());
    if (request->finished() || request->cancelled()) {
      kv_cache_manager_client_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // remove the request from the priority queue
      waiting_priority_queue.pop();
      continue;
    }

    const size_t num_sequences = request->sequences().size();
    if (!request->preempted()) {
      CHECK(num_sequences == 1)
          << "Waiting request should have only one sequence.";
    }

    // TODO: FIXME later
    // Optimization of the scheduling algorithm under multiple sequences
    // TODO: can refactor like handle_decode otherwise request with multiple
    // long sequences may stuck when n>1
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

      prefill_sequence->sync_result();

      size_t num_tokens = prefill_sequence->num_need_compute_tokens();
      if (remaining_token_budget < allocated_tokens + num_tokens ||
          remaining_seq_budget < allocated_seqs + 1) {
        can_schedule = false;
        break;
      }

      // preempt offline decode
      if (!kv_cache_manager_client_->allocate(prefill_sequence.get())) {
        can_schedule = false;
        if (options_.enable_online_preempt_offline() && !request->offline() &&
            !running_queue_offline_->empty()) {
          size_t num_request_to_evict = 0;
          // according to the prefill_sequence num tokens to check if can
          // allocate blocks for it through evict

          bool enough_to_evict =
              check_if_enough_to_evict(running_queue_offline_.get(),
                                       prefill_sequence.get(),
                                       num_tokens,
                                       num_request_to_evict);
          if (enough_to_evict) {
            for (size_t i = 0; i < num_request_to_evict; ++i) {
              std::shared_ptr<Request> request_to_preempt =
                  running_queue_offline_->back();
              ++num_online_prefill_preempt_offline_requests;
              kv_cache_manager_client_->deallocate(request_to_preempt.get());
              running_queue_offline_->pop_back();
              // add preemptable request to waiting priority queue
              // TO IMPROVE?: not process this offline request in current batch
              request_to_preempt->set_preempted();
              waiting_priority_queue_offline_.push(request_to_preempt);
            }
            if (!kv_cache_manager_client_->allocate(prefill_sequence.get())) {
              LOG(ERROR) << "Should be able to allocate after preempting "
                         << num_request_to_evict
                         << " offline requests, but failed.";
              can_schedule = false;
            } else {
              can_schedule = true;
            }
          }
        }
        if (!can_schedule) {
          kv_cache_manager_client_->deallocate(prefill_sequence.get());
          break;
        }
      }
      prefill_sequences_budget.emplace_back(num_tokens);
      prefill_sequences.emplace_back(prefill_sequence.get());
      allocated_tokens += num_tokens;
      allocated_seqs += 1;
    }

    if (!can_schedule) {
      for (auto& seq : prefill_sequences) {
        // release shared blocks
        kv_cache_manager_client_->deallocate(seq);
      }
      break;
    }

    remaining_token_budget -= allocated_tokens;
    remaining_seq_budget -= allocated_seqs;
    waiting_priority_queue.pop();
    running_requests_.emplace_back(request);
    running_sequences_.insert(running_sequences_.end(),
                              prefill_sequences.begin(),
                              prefill_sequences.end());
    running_sequences_budgets_.insert(running_sequences_budgets_.end(),
                                      prefill_sequences_budget.begin(),
                                      prefill_sequences_budget.end());
  }
  // maybe can pre-compute if prompt beyond length
  if (running_sequences_.empty() && !waiting_priority_queue.empty() &&
      running_queue_->empty()) {
    LOG(ERROR) << "Request prompt is too long, no enough memory to schedule "
                  "a single sequence.";
    // no enough memory to schedule single sequence, just finish the request
    std::shared_ptr<Request> request(waiting_priority_queue.top());
    waiting_priority_queue.pop();
    kv_cache_manager_client_->deallocate(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough memory to schedule single sequence"});
  }

  if (!running_sequences_.empty()) {
    last_step_prefill_ = true;
  }
}

void ContinuousScheduler::handle_decode_requests(
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    size_t& num_offline_decode_preempt_offline_requests,
    size_t& num_online_decode_preempt_online_requests,
    size_t& num_online_decode_preempt_offline_requests,
    std::unique_ptr<DecodePriorityQueue>& running_queue) {
  while (!running_queue->empty() &&
         remaining_token_budget > options_.num_speculative_tokens() &&
         remaining_seq_budget > 0) {
    std::shared_ptr<Request> request = running_queue->top();
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
      if (allocated_tokens + options_.num_speculative_tokens() >=
              remaining_token_budget ||
          allocated_seqs >= remaining_seq_budget) {
        has_enough_budget = false;
        break;
      }
      // sequence token already appended
      size_t updated_num_tokens =
          sequence->num_tokens() + options_.num_speculative_tokens();
      // no blocks left
      if (!kv_cache_manager_client_->allocate(sequence.get(),
                                              updated_num_tokens)) {
        has_enough_blocks = false;
        break;
      }

      if (sequence->if_cache_block_for_prefill()) {
        kv_cache_manager_client_->cache(sequence.get());
      }

      // update the allocated tokens for the sequence
      allocated_tokens += options_.num_speculative_tokens() + 1;
      allocated_seqs += 1;
      candidate_sequences.emplace_back(sequence.get());
      candidate_token_budgets.emplace_back(options_.num_speculative_tokens() +
                                           1);
    }
    CHECK(allocated_tokens <= remaining_token_budget);
    CHECK(allocated_seqs <= remaining_seq_budget);

    // schedule candidates in the request if there are enough blocks
    if (has_enough_budget && has_enough_blocks) {
      // remove the request from the priority queue
      running_queue->pop_top();
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
      handle_abnormal_request(running_queue,
                              candidate_sequences,
                              candidate_token_budgets,
                              allocated_tokens,
                              allocated_seqs,
                              remaining_token_budget,
                              remaining_seq_budget,
                              true, /*budget_exhausted*/
                              false /*blocks_exhausted*/);
      break;
    }

    // memory exhausted, try to preempt lowest priority request
    // continue to evict blocks until enough or no other requests that can be
    // preempted
    if (options_.enable_online_preempt_offline() && !request->offline() &&
        !running_queue_offline_->empty()) {
      std::shared_ptr<Request> request_to_preempt =
          running_queue_offline_->back();
      ++num_online_decode_preempt_offline_requests;
      kv_cache_manager_client_->deallocate(request_to_preempt.get());
      running_queue_offline_->pop_back();
      // add preemptable request to waiting priority queue
      request_to_preempt->set_preempted();
      waiting_priority_queue_offline_.push(request_to_preempt);
      continue;
    } else if (running_queue->size() > 1) {
      std::shared_ptr<Request> request_to_preempt = running_queue->back();
      if (request_to_preempt.get() != request.get()) {
        // TO IMPROVE: kv cache offload to cpu
        kv_cache_manager_client_->deallocate(request_to_preempt.get());
        running_queue->pop_back();
        // add preemptable request to waiting priority queue
        request_to_preempt->set_preempted();
        if (request_to_preempt->offline()) {
          ++num_offline_decode_preempt_offline_requests;
          waiting_priority_queue_offline_.push(request_to_preempt);
        } else {
          ++num_online_decode_preempt_online_requests;
          waiting_priority_queue_.push(request_to_preempt);
        }

      } else {
        LOG(FATAL) << "Unexpected error: preempting the candidate itself.";
      }

      continue;
    }

    // no requests left to preempt
    handle_abnormal_request(running_queue,
                            candidate_sequences,
                            candidate_token_budgets,
                            allocated_tokens,
                            allocated_seqs,
                            remaining_token_budget,
                            remaining_seq_budget,
                            false, /*budget_exhausted*/
                            true /*blocks_exhausted*/);
    break;
  }
}

// NOTE: refactor ChunkedPrefillScheduler and ContinuousScheduler later.
void ContinuousScheduler::handle_abnormal_request(
    std::unique_ptr<DecodePriorityQueue>& running_queue,
    const std::vector<Sequence*>& candidate_sequences,
    const std::vector<size_t>& candidate_token_budgets,
    const size_t& allocated_tokens,
    const size_t& allocated_seqs,
    size_t& remaining_token_budget,
    size_t& remaining_seq_budget,
    bool budget_exhausted,
    bool blocks_exhausted) {
  std::shared_ptr<Request> request = running_queue->top();
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
      CHECK(running_queue->size() == 1)
          << "Running queue size is not 1, there maybe a bug of request "
             "preemption logic. running_queue_.size ="
          << running_queue_->size();
      if (!FLAGS_enable_continuous_kvcache &&
          (util::sum(kv_cache_manager_client_->num_used_blocks()) !=
           request->total_num_blocks())) {
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
    running_queue_->pop_top();
    kv_cache_manager_client_->deallocate(request.get());
    response_processor_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED,
         "No enough resource to schedule a single sequence"});
  } else {
    // partially schedule the sequences in request
    running_queue->pop_top();
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

void ContinuousScheduler::handle_running_requests(
    std::shared_ptr<Request> request) {
  if (request->finished() || request->cancelled()) {
    LOG(FATAL) << "Unknow error, finished/cancelled request have be handled "
                  "before. request_id is "
               << request->request_id();
  }

  // check if the request can be expanded
  if (request->expand_sequences()) {
    // cache the blocks to share among the sequences
    kv_cache_manager_client_->cache(request->sequences()[0].get());
  }

  // release blocks for finished sequences here
  for (auto& sequence : request->sequences()) {
    if (sequence->finished()) {
      kv_cache_manager_client_->deallocate(sequence.get());
    }
  }
}

std::vector<Batch> ContinuousScheduler::prepare_batch() {
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
      if (request->offline()) {
        waiting_priority_queue_offline_.push(request);
      } else {
        waiting_priority_queue_.push(request);
      }
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
      kv_cache_manager_client_->deallocate(request.get());
      // release the ownership of the request
      finished_requests.emplace_back(request);
      // finished request is set to nullptr
      *it = nullptr;
    }
  }

  if (options_.priority_strategy() == "FCFS") {
    if (last_step_prefill_) {
      // insert all requests to the back of running_queue_
      // 1. last step is prefill step:
      // new prefill has high priority, but these requests has lower priority
      // then existed requests in running_queue_ in decoding stage.
      // so we need to push them to the back of running_queue_.
      for (auto it = running_requests_.begin(); it != running_requests_.end();
           ++it) {
        // finished request is set to nullptr
        if (*it == nullptr) {
          continue;
        }
        handle_running_requests(*it);
        if ((*it)->offline()) {
          running_queue_offline_->push(*it, last_step_prefill_);
        } else {
          running_queue_->push(*it, last_step_prefill_);
        }
      }
    } else {
      // insert all requests to the front of running_queue_
      // 2. last step is decode step:
      // We need to traverse running_requests_ array in reverse order.
      // Because there may be some unexecuted requests with
      // lower priorities remaining in the running_queue_.
      // For the requests in running_requests_,
      // their priorities are all higher than those of the
      // remaining requests. Therefore, the `push_front`
      // method needs to be used.
      //
      for (auto it = running_requests_.rbegin(); it != running_requests_.rend();
           ++it) {
        // finished request is set to nullptr
        if (*it == nullptr) {
          continue;
        }
        handle_running_requests(*it);
        if ((*it)->offline()) {
          running_queue_offline_->push(*it, last_step_prefill_);
        } else {
          running_queue_->push(*it, last_step_prefill_);
        }
      }
    }
  } else {
    // directly push running requests to the priority queue
    for (auto it = running_requests_.begin(); it != running_requests_.end();
         ++it) {
      if (*it == nullptr) {
        continue;
      }
      handle_running_requests(*it);
      if ((*it)->offline()) {
        running_queue_offline_->push(*it);
      } else {
        running_queue_->push(*it);
      }
    }
  }

  // clear previous batch
  last_step_prefill_ = false;
  running_requests_.clear();
  running_sequences_.clear();
  running_sequences_budgets_.clear();

  // remaining budget for the current batch
  size_t remaining_token_budget = options_.max_tokens_per_batch();
  size_t remaining_seq_budget = std::max(options_.max_seqs_per_batch(), 1);
  size_t num_preempted_requests = 0;
  size_t num_offline_decode_preempt_offline_requests = 0;
  size_t num_online_decode_preempt_online_requests = 0;
  size_t num_online_prefill_preempt_offline_requests = 0;
  size_t num_online_decode_preempt_offline_requests = 0;
  // TO IMPROVE?: handle online decode request before prefill offline request
  handle_prefill_requests(remaining_token_budget,
                          remaining_seq_budget,
                          waiting_priority_queue_,
                          num_online_prefill_preempt_offline_requests,
                          finished_requests);
  handle_prefill_requests(remaining_token_budget,
                          remaining_seq_budget,
                          waiting_priority_queue_offline_,
                          num_online_prefill_preempt_offline_requests,
                          finished_requests);

  if (running_sequences_.empty()) {
    // Handle decoding requests.
    // no prefill request, schedule the decode requests in the running priority
    // queue
    handle_decode_requests(remaining_token_budget,
                           remaining_seq_budget,
                           num_offline_decode_preempt_offline_requests,
                           num_online_decode_preempt_online_requests,
                           num_online_decode_preempt_offline_requests,
                           running_queue_);
    handle_decode_requests(remaining_token_budget,
                           remaining_seq_budget,
                           num_offline_decode_preempt_offline_requests,
                           num_online_decode_preempt_online_requests,
                           num_online_decode_preempt_offline_requests,
                           running_queue_offline_);
  }

  num_preempted_requests = num_offline_decode_preempt_offline_requests +
                           num_online_decode_preempt_online_requests +
                           num_online_decode_preempt_offline_requests +
                           num_online_prefill_preempt_offline_requests;
  if (!finished_requests.empty()) {
    response_processor_->process_completed_requests(finished_requests);
  }

  auto batches =
      BatchFactory::get_instance(options_.dp_size())
          ->create_batches(
              running_sequences_,
              running_sequences_budgets_,
              kv_cache_manager_client_->get_copy_in_cache_block_infos(),
              kv_cache_manager_client_->get_copy_out_cache_block_infos());

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
  GAUGE_ADD(num_offline_decode_preempt_offline_requests,
            num_offline_decode_preempt_offline_requests);
  GAUGE_ADD(num_online_decode_preempt_online_requests,
            num_online_decode_preempt_online_requests);
  GAUGE_ADD(num_online_prefill_preempt_offline_requests,
            num_online_prefill_preempt_offline_requests);
  GAUGE_ADD(num_online_decode_preempt_offline_requests,
            num_online_decode_preempt_offline_requests);

  GAUGE_SET(num_running_sequences, running_sequences_.size());

  GAUGE_SET(kv_cache_utilization_perc,
            kv_cache_manager_client_->kv_cache_utilization());
  if (!FLAGS_enable_continuous_kvcache) {
    GAUGE_SET(
        num_blocks_in_prefix_cache,
        util::min(kv_cache_manager_client_->num_blocks_in_prefix_cache()));
    GAUGE_SET(num_free_blocks,
              util::max(kv_cache_manager_client_->num_free_blocks()));
    GAUGE_SET(num_used_blocks,
              util::min(kv_cache_manager_client_->num_used_blocks()));
  }
  return batches;
}

std::vector<Batch> ContinuousScheduler::schedule_request(
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

    if (!waiting_priority_queue_.empty() || !running_queue_->empty() ||
        !waiting_priority_queue_offline_.empty() ||
        !running_queue_offline_->empty()) {
      continue;
    }

    const auto now = absl::Now();
    if (now > deadline) {
      break;
    }
    // wait for new requests to arrive
    constexpr uint64_t kStepSleepTimeMs = 10;
    const auto time_to_sleep =
        std::min(absl::Milliseconds(kStepSleepTimeMs), deadline - now);
    absl::SleepFor(time_to_sleep);
  }
  // return an empty batch
  return batch;
}

void ContinuousScheduler::prepare_cache_async(
    std::shared_ptr<Request>& request) {
  if (request->sequences()[0]->kv_state().num_kv_blocks() != 0 ||
      request->sequences()[0]->host_kv_state().num_kv_blocks() != 0) {
    return;
  }
  for (auto& prefill_sequence : request->sequences()) {
    const size_t num_additional_blocks =
        kv_cache_manager_client_->pre_allocate(prefill_sequence.get());
    if (num_additional_blocks > 0) {
      const auto host_blocks = prefill_sequence->host_kv_state().kv_blocks();
      std::vector<CacheBlockInfo> contents;
      contents.reserve(num_additional_blocks);
      for (int i = host_blocks.size() - num_additional_blocks;
           i < host_blocks.size();
           i++) {
        contents.emplace_back(
            -1, host_blocks[i].id(), host_blocks[i].get_immutable_hash_value());
      }

      auto futures = engine_->load_kv_blocks_from_store_async(
          prefill_sequence->dp_rank(), std::move(contents));
      prefill_sequence->set_async_result(std::move(futures));
    }
  }
}

// step the scheduler forward by one step
// may get blocked if there are no requests to process
void ContinuousScheduler::step(const absl::Duration& timeout) {
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
    kv_cache_manager_client_->reset_copy_content();
    // process request output in batch
    process_batch_output(false);
  } else {
    step_with_schedule_overlap(timeout);
  }
}

void ContinuousScheduler::step_with_schedule_overlap(
    const absl::Duration& timeout) {
  // get a new batch of requests
  std::vector<Batch> batch = schedule_request(timeout);
  bool cur_batch_all_empty =
      std::all_of(batch.begin(), batch.end(), [](const Batch& one_batch) {
        return one_batch.empty();
      });
  bool last_batch_all_empty = std::all_of(
      last_batch_.begin(), last_batch_.end(), [](const Batch& one_batch) {
        return one_batch.empty();
      });
  if (cur_batch_all_empty && last_batch_all_empty) {
    return;
  }

  if (!cur_batch_all_empty) {
    engine_->step(batch);
    kv_cache_manager_client_->reset_copy_content();
  }

  // producer-consumer mode, make sure only one step is scheduled in advance
  if (!is_first_step_ && !last_batch_all_empty) {
    engine_->update_last_step_result(last_batch_);
    process_batch_output(true);
  }
  last_batch_ = std::move(batch);
  last_running_sequences_ = running_sequences_;
  last_running_requests_ = running_requests_;
  is_first_step_ = false;
}

void ContinuousScheduler::generate() {
  bool batch_empty = true;
  while (num_pending_requests() > 0 || !batch_empty) {
    // build a batch of requests/sequences
    auto batch = prepare_batch();
    batch_empty = true;
    for (auto& b : batch) {
      batch_empty &= b.empty();
    }
    if (batch_empty) {
      continue;
    }

    // run inference for the batch
    engine_->step(batch);
    kv_cache_manager_client_->reset_copy_content();

    // process request output in batch
    process_batch_output(false);
  }

  // wait for all responses done
  response_processor_->wait_completion();
}

void ContinuousScheduler::update_token_latency_metrics(
    std::vector<Sequence*>& sequences) {
  const auto now = absl::Now();
  for (Sequence* sequence : sequences) {
    int64_t tbt_milliseconds = sequence->tbt(now);
    if (sequence->is_first_token()) {
      HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                        tbt_milliseconds);
      sequence->set_time_to_first_token_latency_seconds(
          static_cast<double>(tbt_milliseconds) / 1000);
    } else {
      HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
    }
  }
}

void ContinuousScheduler::process_batch_output(bool enable_schedule_overlap) {
  std::vector<Sequence*>& to_be_processed_sequences =
      enable_schedule_overlap ? last_running_sequences_ : running_sequences_;
  std::vector<std::shared_ptr<Request>>& to_be_processed_requests =
      enable_schedule_overlap ? last_running_requests_ : running_requests_;
  // update token latency metrics
  update_token_latency_metrics(to_be_processed_sequences);

  // update slot usage and activation metrics
  if (!FLAGS_enable_continuous_kvcache) {
    update_memory_metrics(to_be_processed_sequences);
  }

  std::vector<std::shared_ptr<Request>> stream_requests;
  // process request output in batch
  for (auto request : to_be_processed_requests) {
    // ignore cancelled/finished requests when enable_schedule_overlap.
    if (options_.enable_schedule_overlap() && request->state().stream) {
      if (!request->finished() && !request->cancelled()) {
        stream_requests.emplace_back(request);
      }
      // handle token when last token not be handled.
      if (request->finished() && !request->last_token_handled()) {
        request->handle_last_token();
        stream_requests.emplace_back(request);
      }
    } else if (request->state().stream) {
      stream_requests.emplace_back(request);
    }
  }
  if (!stream_requests.empty()) {
    response_processor_->process_stream_requests(stream_requests);
  }
}

std::vector<Block> ContinuousScheduler::allocate_blocks_for(size_t token_num,
                                                            int32_t& dp_rank) {
  return kv_cache_manager_client_->allocate(token_num, dp_rank);
}

std::vector<int64_t> ContinuousScheduler::get_num_occupied_slots(
    std::vector<Sequence*>& sequences) const {
  std::vector<int64_t> num_occupied_slots(options_.dp_size());
  std::vector<int64_t> num_unfilled_blocks(options_.dp_size());
  std::vector<size_t> num_used_blocks =
      kv_cache_manager_client_->num_used_blocks();

  const int block_size = kv_cache_manager_client_->options().block_size();

  for (auto& sequence : sequences) {
    const int32_t dp_rank = sequence->dp_rank();
    // last_block_len is the length of the last unfilled block of each sequence.
    int32_t last_block_len =
        sequence->kv_state().kv_cache_tokens_num() % block_size;
    num_occupied_slots[dp_rank] += last_block_len;
    num_unfilled_blocks[dp_rank] += last_block_len > 0 ? 1 : 0;
  }

  for (int32_t dp_rank = 0; dp_rank < options_.dp_size(); ++dp_rank) {
    num_occupied_slots[dp_rank] +=
        (num_used_blocks[dp_rank] - num_unfilled_blocks[dp_rank]) * block_size;
  }
  return num_occupied_slots;
}

std::vector<int64_t> ContinuousScheduler::get_active_activation_in_bytes() {
  std::vector<int64_t> all_active_activation_in_bytes =
      engine_->get_active_activation_memory();
  std::vector<int64_t> active_activation_in_bytes(options_.dp_size());

  const int32_t dp_local_tp_size =
      all_active_activation_in_bytes.size() / options_.dp_size();

  for (int32_t dp_rank = 0; dp_rank < options_.dp_size(); ++dp_rank) {
    active_activation_in_bytes[dp_rank] =
        all_active_activation_in_bytes[dp_rank * dp_local_tp_size];
  }
  return active_activation_in_bytes;
}

void ContinuousScheduler::update_memory_metrics(
    std::vector<Sequence*>& sequences) {
  std::vector<int64_t> num_occupied_slots = get_num_occupied_slots(sequences);
  std::vector<int64_t> active_activation_size_in_bytes =
      get_active_activation_in_bytes();
  int64_t num_total_slots = kv_cache_manager_client_->options().num_blocks() *
                            kv_cache_manager_client_->options().block_size();

  for (int32_t dp_rank = 0; dp_rank < options_.dp_size(); ++dp_rank) {
    double occupied_slots_ratio =
        static_cast<double>(num_occupied_slots[dp_rank]) / num_total_slots;
    double active_kv_cache_size_in_kilobytes =
        occupied_slots_ratio * GAUGE_VALUE(total_kv_cache_size_in_kilobytes);
    int64_t active_activation_size_in_kilobytes =
        active_activation_size_in_bytes[dp_rank] / 1024;

    MULTI_HISTOGRAM_OBSERVE(
        active_kv_cache_size_in_kilobytes,
        std::to_string(dp_rank),
        static_cast<int64_t>(active_kv_cache_size_in_kilobytes));

    if (FLAGS_enable_chunked_prefill) {
      MULTI_HISTOGRAM_OBSERVE(decode_active_activation_size_in_kilobytes,
                              std::to_string(dp_rank),
                              active_activation_size_in_kilobytes);
    } else {
      if (sequences[0]->is_first_token()) {
        MULTI_HISTOGRAM_OBSERVE(prefill_active_activation_size_in_kilobytes,
                                std::to_string(dp_rank),
                                active_activation_size_in_kilobytes);
      } else {
        MULTI_HISTOGRAM_OBSERVE(decode_active_activation_size_in_kilobytes,
                                std::to_string(dp_rank),
                                active_activation_size_in_kilobytes);
      }
    }
  }
}
}  // namespace xllm
