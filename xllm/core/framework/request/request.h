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

enum class Urgency { URGENT = 2, TIMEOUT = 1, NORMAL = 0 };

class Request : public RequestBase {
 public:
  Request(const std::string& request_id,
          const std::string& x_request_id,
          const std::string& x_request_time,
          const RequestState& state,
          const std::string& service_request_id = "",
          bool offline = false,
          int32_t ttlt_slo_ms = 0,
          RequestPriority priority = RequestPriority::NORMAL,
          int32_t ttft_slo_ms = 0,
          int32_t tpot_slo_ms = 0,
          int32_t tpot_priority_weight = 1,
          int32_t ttft_priority_weight = 1,
          int32_t ttlt_priority_weight = 1,
          int32_t priority_weight = 1);

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

  absl::Time created_time() const { return created_time_; }

  int32_t get_deadline_ms() const { return deadline_ms_; }

  void set_deadline_ms() {
    auto& sequence = sequences()[0];
    // w/o first token buffer
    if (sequence->is_prefill_stage()) {
      deadline_ms_ = ttft_slo_ms_;
    } else {
      deadline_ms_ =
          std::min(static_cast<int32_t>(
                       sequence->time_to_first_token_latency_seconds() * 1000),
                   ttft_slo_ms_) +
          (sequence->num_tokens() - sequence->num_prompt_tokens()) *
              tpot_slo_ms_;
    }
    // w/ first token buffer
    // deadline_ms_ = ttft_slo_ms_ + (sequence->num_tokens() -
    // sequence->num_prompt_tokens()) * tpot_slo_ms_;
  }

  int32_t get_remaining_time() const {
    return get_deadline_ms() - get_elapsed_time_ms();
  }

  void set_elapsed_time_ms() {
    elapsed_time_ms_ = static_cast<int32_t>(
        absl::ToDoubleSeconds(absl::Now() - created_time_) * 1000);
  }
  int32_t get_elapsed_time_ms() const { return elapsed_time_ms_; }

  const bool offline() const { return offline_; }
  const RequestPriority priority() const { return priority_; }
  const int32_t ttlt_slo_ms() const { return ttlt_slo_ms_; }
  const int32_t ttft_slo_ms() const { return ttft_slo_ms_; }
  const int32_t tpot_slo_ms() const { return tpot_slo_ms_; }
  const int32_t tpot_priority_weight() const { return tpot_priority_weight_; }
  const int32_t ttft_priority_weight() const { return ttft_priority_weight_; }
  const int32_t ttlt_priority_weight() const { return ttlt_priority_weight_; }

  void set_urgency(Urgency urgency) { urgency_ = urgency; }
  Urgency urgency() const { return urgency_; }

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

  int32_t elapsed_time_ms_ = 0;

  int32_t deadline_ms_ = 0;

  bool offline_;

  int32_t ttlt_slo_ms_;

  int32_t ttft_slo_ms_;

  int32_t tpot_slo_ms_;

  int32_t tpot_priority_weight_;

  int32_t ttft_priority_weight_;

  int32_t ttlt_priority_weight_;

  int32_t priority_weight_;

  RequestPriority priority_;

  Urgency urgency_ = Urgency::NORMAL;

 private:
  void create_sequences_group();
};

}  // namespace xllm
