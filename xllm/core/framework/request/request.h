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

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "common.pb.h"
#include "request_base.h"
#include "request_state.h"
#include "sequences_group.h"
#include "stopping_checker.h"
#include "util/threadpool.h"

namespace xllm {

enum class RequestPriority { DEFAULT = 0, HIGH = 1, NORMAL = 2, LOW = 3 };

class Request : public RequestBase {
 public:
  Request(const std::string& request_id,
          const std::string& x_request_id,
          const std::string& x_request_time,
          const RequestState& state,
          const std::string& service_request_id = "",
          bool offline = false,
          int32_t slo_ms = 0,
          RequestPriority priority = RequestPriority::NORMAL);

  bool finished() const;

  std::vector<std::unique_ptr<Sequence>>& sequences() {
    return sequences_group_->sequences();
  }
  bool expand_sequences(bool share_prefix = true);

  SequencesGroup* sequence_group() { return sequences_group_.get(); }

  void set_cancel();

  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

  // Get the elapsed time since the request was created.
  double elapsed_seconds() const {
    return absl::ToDoubleSeconds(absl::Now() - created_time_);
  }

  RequestOutput generate_output(const Tokenizer& tokenizer,
                                ThreadPool* thread_pool = nullptr);

  void handle_last_token();

  bool last_token_handled() const { return state_.handle_last_token_done; }

  size_t total_num_blocks();

  void set_preempted() { state_.preempted = true; }

  bool preempted() const { return state_.preempted; }

  void log_statistic(double total_latency);

  void log_error_statistic(Status status);

  const bool offline() const { return offline_; }
  const int32_t slo_ms() const { return slo_ms_; }
  const RequestPriority priority() const { return priority_; }

  RequestState& state() { return state_; }
  void update_connection_status();

  bool check_beam_search() const {
    return state_.sampling_param.beam_width > 1;
  }

  bool is_chunked_prefill_stage() const {
    return sequences_group_->is_chunked_prefill_stage();
  }

 private:
  RequestState state_;
  // list of sequences to generate completions for the prompt
  // use deque instead of vector to avoid no-copy move for Sequence
  //  std::deque<Sequence> sequences;
  std::unique_ptr<SequencesGroup> sequences_group_;

  std::atomic<bool> cancelled_{false};

  bool offline_;

  int32_t slo_ms_;

  RequestPriority priority_;

 private:
  void create_sequences_group();
};

}  // namespace xllm
