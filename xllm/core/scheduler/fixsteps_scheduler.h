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
#include <folly/futures/Future.h>

#include <limits>
#include <memory>
#include <queue>

#include "async_response_processor.h"
#include "common/macros.h"
#include "common/types.h"
#include "framework/batch/batch.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "runtime/xservice_client.h"
#include "scheduler.h"
#include "scheduler/continuous_scheduler.h"

namespace xllm {
class Engine;

class FixStepsScheduler final : public ContinuousScheduler {
 public:
  FixStepsScheduler(Engine* engine, const Options& options);
  virtual ~FixStepsScheduler() = default;

  bool add_request(std::shared_ptr<Request>& request) override;

  // step the scheduler forward by one step
  // may get blocked if there are no requests to process
  void step(const absl::Duration& timeout) override;

 private:
  std::vector<Batch> schedule_request(const absl::Duration& timeout);

  // build a batch of requests from the priority queue
  virtual std::vector<Batch> prepare_batch();

  void handle_prefill_requests(
      size_t& remaining_token_budget,
      size_t& remaining_seq_budget,
      std::vector<std::shared_ptr<Request>>& finished_requests);
};

}  // namespace xllm
