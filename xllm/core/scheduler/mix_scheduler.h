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

#include <algorithm>
#include <list>
#include <memory>
#include <queue>

#include "scheduler/chunked_prefill_scheduler.h"
#include "scheduler/continuous_scheduler.h"

namespace xllm {

// MixScheduler does not explicitly specify whether decoding or prefilling takes
// priority; instead, it mixes prefilling and decoding requests in a single
// queue. Currently, it's only for multi-priority scheduling algorithm ProSched.
class MixScheduler : public ChunkedPrefillScheduler {
 public:
  MixScheduler(Engine* engine, const Options& options);
  virtual ~MixScheduler();

 protected:
  std::list<std::shared_ptr<Request>> running_queue_;

  // build a batch of requests from the priority queue
  virtual std::vector<Batch> prepare_batch() override;

  virtual bool if_queue_not_empty() override;

 private:
  void handle_running_queue_requests(
      double& latency_budget,
      double& estimate_latency,
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      size_t& num_preempted_requests,
      std::vector<Sequence*>& prefill_stage_sequences,
      std::list<std::shared_ptr<Request>>& running_queue,
      bool& budget_exhausted,
      bool& blocks_exhausted);

  void get_latency_budget_and_request_order(
      std::list<std::shared_ptr<Request>>& running_queue,
      double& latency_budget);
  int32_t get_max_chunk(Sequence* sequence,
                        int32_t latency_budget,
                        bool use_quadratic_formula = false);
};

}  // namespace xllm
