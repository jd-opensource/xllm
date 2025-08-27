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

#pragma once

#include <absl/time/time.h>
#include <folly/MPMCQueue.h>

#include <limits>
#include <memory>
#include <queue>

#include "async_response_processor.h"
#include "common/macros.h"
#include "common/types.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/xservice_client.h"
#include "scheduler.h"

namespace xllm {
class Engine;

struct RequestComparator {
  bool operator()(std::shared_ptr<Request> a,
                  std::shared_ptr<Request> b) const {
    return a->created_time() > b->created_time();
  }
};

class ContinuousScheduler : public Scheduler {
 public:
  struct Options {
    // the maximum number of tokens per batch
    PROPERTY(int32_t,
             max_tokens_per_batch) = std::numeric_limits<int32_t>::max();

    // the maximum number of sequences per batch
    PROPERTY(int32_t, max_seqs_per_batch) = 256;

    // the max tokens per chunk for request in prefill stage.
    PROPERTY(int32_t, max_tokens_per_chunk_for_prefill) = 2048;

    // the number of speculative tokens per step
    PROPERTY(int32_t, num_speculative_tokens) = 0;

    // the number of speculative tokens per step
    PROPERTY(int32_t, dp_size) = 1;

    // enable disaggregated PD mode.
    PROPERTY(bool, enable_disagg_pd) = false;

    // enable decode response to service directly
    PROPERTY(bool, enable_decode_response_to_service) = false;

    PROPERTY(std::optional<InstanceRole>, instance_role);

    PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

    // In general decode instance send a batch responses to prefill in disagg pd
    // mode. here, we add a flag to control whether send a batch or single
    // response once, This will help us to debug code. default value is false.
    PROPERTY(bool, enable_batch_response) = false;

    // support P send batch reqs to D.
    // max_reqs_p2d_once represents the maximum number
    // of requests that can be sent once.
    // default value is 1.
    PROPERTY(int32_t, max_reqs_p2d_once) = 1;

    PROPERTY(bool, enable_schedule_overlap) = false;

    PROPERTY(bool, enable_chunked_prefill) = true;

    PROPERTY(bool, enable_service_routing) = false;
  };

  ContinuousScheduler(Engine* engine, const Options& options);
  virtual ~ContinuousScheduler();

  bool add_request(std::shared_ptr<Request>& request) override;

  void step(const absl::Duration& timeout) override;

  void generate() override;

  // inc/dec pending requests
  void incr_pending_requests(size_t count) override {
    pending_requests_.fetch_add(count, std::memory_order_relaxed);
  }
  void decr_pending_requests() override {
    const auto old_value =
        pending_requests_.fetch_sub(1, std::memory_order_relaxed);
    CHECK_GT(old_value, 0) << "pending requests underflow";
  }

  size_t num_pending_requests() {
    return pending_requests_.load(std::memory_order_relaxed);
  }

  virtual uint32_t get_waiting_requests_num() const override {
    return waiting_priority_queue_.size();
  }

 protected:
  // allocate actual token_num slots.
  std::vector<Block> allocate_blocks_for(size_t token_num, int32_t& dp_rank);

  const Options options_;

  // the engine to run the batch
  Engine* engine_;

  // the block manager to manage the cache blocks
  BlockManagerPool* block_manager_;

  // a thread safe queue of requests, bounded by kRequestQueueSize
  // the schedule owns the requests and manages their lifetimes.
  folly::MPMCQueue<std::shared_ptr<Request>> request_queue_;

  // a batch of requests in running state, sorted by priority from high to low.
  // This may include decoding requests and prefill requests in chunked prefill
  // scheudler.
  std::vector<std::shared_ptr<Request>> running_requests_;

  // a batch of sequences that scheduled to run, sorted by priority from high to
  std::vector<Sequence*> running_sequences_;

  // token budget for each running sequence
  std::vector<size_t> running_sequences_budgets_;

  // preemptable requests that hold cache slots, sorted by priority from high to
  // low.
  std::deque<std::shared_ptr<Request>> preemptable_requests_;

  std::unique_ptr<AsyncResponseProcessor> response_processor_;

  bool enable_prefix_cache_ = false;

  // the number of requests that are waiting to be scheduled
  std::atomic<size_t> pending_requests_{0};

  // Requests with HIGH priority are processed first, followed by MEDIUM
  // priority requests, and finally LOW priority requests. Within each priority
  // level, requests are handled on First-Come-First-Served (FCFS) basis.
  using RequestPriorityQueue =
      std::priority_queue<std::shared_ptr<Request>,
                          std::vector<std::shared_ptr<Request>>,
                          RequestComparator>;
  // keep all new requests, generally speaking, they do not have any kv cache.
  RequestPriorityQueue waiting_priority_queue_;

  // keep all running request from high priority to low.
  // NOTE: Maybe not all requests are scheduled in one step,
  // this is decided by the kv blocks usage.
  // This contain all decoding requests and requests that have been
  // popped from waiting_priority_queue_ but remain in prefill stage,
  // these requests have already allocated some kv caches,
  // so they can be preemeted in scheduler.
  std::deque<std::shared_ptr<Request>> running_queue_;

  // is last step handle prefill requests
  bool last_step_prefill_ = false;

  void handle_abnormal_request(
      const std::vector<Sequence*>& candidate_sequences,
      const std::vector<size_t>& candidate_token_budgets,
      const size_t& allocated_tokens,
      const size_t& allocated_seqs,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      bool budget_exhausted,
      bool block_exhausted);
  void handle_running_requests(std::shared_ptr<Request> request);

  // build a batch of requests from the priority queue
  virtual std::vector<Batch> prepare_batch();

  virtual void handle_prefill_requests(
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& finished_requests);

  void handle_decode_requests(size_t& remaining_token_budget,
                              size_t& remaining_seq_budget,
                              size_t& num_preempted_requests);

 private:
  std::vector<Batch> schedule_request(const absl::Duration& timeout);

  // process the batch output
  void process_batch_output(bool enable_schedule_overlap);

  void step_with_schedule_overlap(const absl::Duration& timeout);

  std::vector<int64_t> get_num_occupied_slots(
      std::vector<Sequence*>& sequences) const;
  std::vector<int64_t> get_active_activation_in_bytes();

 private:
  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // params for enable_schedule_overlap case
  std::vector<Batch> last_batch_;
  std::vector<std::shared_ptr<Request>> last_running_requests_;
  std::vector<Sequence*> last_running_sequences_;
  bool is_first_step_ = true;
};

}  // namespace xllm
