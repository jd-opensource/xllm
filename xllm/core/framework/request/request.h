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
#include "request_state.h"
#include "sequences_group.h"
#include "stopping_checker.h"

namespace xllm {

class Request {
 public:
  Request(const std::string& request_id,
          const std::string& x_request_id,
          const std::string& x_request_time,
          const RequestState& state,
          const std::string& service_request_id = "");

  bool finished() const;

  std::vector<std::unique_ptr<Sequence>>& sequences() {
    return sequences_group_->sequences();
  }
  bool expand_sequences(bool share_prefix = true);

  void set_cancel() { cancelled_.store(true, std::memory_order_relaxed); }

  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

  // Get the elapsed time since the request was created.
  double elapsed_seconds() const {
    return absl::ToDoubleSeconds(absl::Now() - created_time_);
  }

  RequestOutput generate_output(const Tokenizer& tokenizer);

  void handle_last_token() { state_.handle_last_token_done = true; }

  bool last_token_handled() const { return state_.handle_last_token_done; }

  size_t total_num_blocks();

  void set_preempted() { state_.preempted = true; }

  bool preempted() const { return state_.preempted; }

  void log_statistic(double total_latency);

  void log_error_statistic(Status status);

  absl::Time created_time() const { return created_time_; }

  const std::string& request_id() const { return request_id_; }

  const std::string& service_request_id() const { return service_request_id_; }

  const std::string& x_request_id() const { return x_request_id_; }

  const std::string& x_request_time() const { return x_request_time_; }

  RequestState& state() { return state_; }

  void update_connection_status();

 private:
  // request create time
  absl::Time created_time_;

  std::string request_id_;

  std::string service_request_id_;

  // x-request-id header value from client
  std::string x_request_id_;

  // x-request-time header value from client
  std::string x_request_time_;

  RequestState state_;

  // list of sequences to generate completions for the prompt
  // use deque instead of vector to avoid no-copy move for Sequence
  //  std::deque<Sequence> sequences;
  std::unique_ptr<SequencesGroup> sequences_group_;

  std::atomic<bool> cancelled_{false};

 private:
  void create_sequences_group();
};

}  // namespace xllm
