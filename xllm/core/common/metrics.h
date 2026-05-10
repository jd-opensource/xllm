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

#include <bvar/bvar.h>
#include <bvar/multi_dimension.h>
#include <bvar/window.h>

#include "common/macros.h"
#include "util/timer.h"

namespace xllm {

using bvar::Adder;

class AutoCounter final {
 public:
  AutoCounter(bvar::Adder<double>& counter) : counter_(counter) {}

  ~AutoCounter() {
    // add the elapsed time to the counter
    counter_ << timer_.elapsed_seconds();
  }

 private:
  // NOLINTNEXTLINE
  bvar::Adder<double>& counter_;

  // the timer
  Timer timer_;
};

}  // namespace xllm

// define helpful macros to hide boilerplate code
// NOLINTBEGIN(bugprone-macro-parentheses)

// define gauge (using bvar::Status for single values)
#define DEFINE_GAUGE(name, desc) bvar::Status<double> GAUGE_##name(#name, 0.0);

#define GAUGE_SET(name, value) GAUGE_##name.set_value(value);

#define GAUGE_ADD(name, value) \
  GAUGE_##name.set_value(GAUGE_##name.get_value() + value);

#define GAUGE_INC(name) GAUGE_##name.set_value(GAUGE_##name.get_value() + 1);

#define GAUGE_VALUE(name) GAUGE_##name.get_value();

// define counter (using bvar::Adder for accumulating values)
#define DEFINE_COUNTER(name, desc) bvar::Adder<double> COUNTER_##name(#name);

#define COUNTER_ADD(name, value) COUNTER_##name << (value);

#define COUNTER_INC(name) COUNTER_##name << 1;

// Counter with per-minute window (sliding window of last 60 seconds).
// Usage: add values same as COUNTER; when read, returns sum in last 60 seconds.
#define DEFINE_COUNTER_PER_MINUTE(name, desc) \
  bvar::WindowEx<bvar::Adder<int64_t>, 60> COUNTER_PER_MINUTE_##name(#name);

#define COUNTER_PER_MINUTE_ADD(name, value) \
  COUNTER_PER_MINUTE_##name << (value);

#define COUNTER_PER_MINUTE_INC(name) COUNTER_PER_MINUTE_##name << 1;

#define DECLARE_COUNTER_PER_MINUTE(name) \
  extern bvar::WindowEx<bvar::Adder<int64_t>, 60> COUNTER_PER_MINUTE_##name;

// Declares a latency counter having a variable name based on line number.
// example: AUTO_COUNTER(a_counter_name);
#define AUTO_COUNTER(name) \
  xllm::AutoCounter SAFE_CONCAT(name, __LINE__)(COUNTER_##name);

// define histogram (using bvar::LatencyRecorder for latency measurements)
#define DEFINE_HISTOGRAM(name, desc) \
  bvar::LatencyRecorder HISTOGRAM_##name(#name);

// value must be int64_t
#define HISTOGRAM_OBSERVE(name, value) HISTOGRAM_##name << (value);

// define multi histogram (using bvar::MultiDimension for multi-dimensional
// measurements)
#define DEFINE_MULTI_HISTOGRAM(name, label, desc)                     \
  bvar::MultiDimension<bvar::LatencyRecorder> MULTI_HISTOGRAM_##name( \
      #name, {(label)});

#define MULTI_HISTOGRAM_OBSERVE(name, key, value)  \
  bvar::LatencyRecorder* latency_recorder_##name = \
      MULTI_HISTOGRAM_##name.get_stats({(key)});   \
  if (latency_recorder_##name) {                   \
    *latency_recorder_##name << (value);           \
  }

// Multi-dimension counter (per-key Adder for failure distribution by instance)
#define DEFINE_MULTI_COUNTER(name, label, desc)                         \
  bvar::MultiDimension<bvar::Adder<double>> MULTI_COUNTER_##name(#name, \
                                                                 {(label)});

#define MULTI_COUNTER_INC(name, key)             \
  do {                                           \
    bvar::Adder<double>* adder_##name =          \
        MULTI_COUNTER_##name.get_stats({(key)}); \
    if (adder_##name) {                          \
      *adder_##name << 1;                        \
    }                                            \
  } while (0)

#define DECLARE_MULTI_COUNTER(name) \
  extern bvar::MultiDimension<bvar::Adder<double>> MULTI_COUNTER_##name;

// declare gauge
#define DECLARE_GAUGE(name) extern bvar::Status<double> GAUGE_##name;

// declare counter
#define DECLARE_COUNTER(name) extern bvar::Adder<double> COUNTER_##name;

// declare histogram
#define DECLARE_HISTOGRAM(name) extern bvar::LatencyRecorder HISTOGRAM_##name;

// declare multi histogram
#define DECLARE_MULTI_HISTOGRAM(name) \
  extern bvar::MultiDimension<bvar::LatencyRecorder> MULTI_HISTOGRAM_##name;

// NOLINTEND(bugprone-macro-parentheses)

// total number of request status
DECLARE_COUNTER(request_status_total_ok);
DECLARE_COUNTER(request_status_total_cancelled);
DECLARE_COUNTER(request_status_total_unknown);
DECLARE_COUNTER(request_status_total_invalid_argument);
DECLARE_COUNTER(request_status_total_deadline_exceeded);
DECLARE_COUNTER(request_status_total_resource_exhausted);
DECLARE_COUNTER(request_status_total_unauthenticated);
DECLARE_COUNTER(request_status_total_unavailable);
DECLARE_COUNTER(request_status_total_unimplemented);

// latency of request handling in seconds
DECLARE_COUNTER(request_handling_latency_seconds_chat);
DECLARE_COUNTER(request_handling_latency_seconds_completion);
DECLARE_COUNTER(tokenization_latency_seconds);
DECLARE_COUNTER(chat_template_latency_seconds);

// latency of prefix cache operations in seconds
DECLARE_COUNTER(prefix_cache_latency_seconds_insert);
DECLARE_COUNTER(prefix_cache_latency_seconds_match);
DECLARE_COUNTER(prefix_cache_latency_seconds_evict);
DECLARE_COUNTER(prefix_cache_match_length_total);
DECLARE_COUNTER(allocate_blocks_latency_seconds);

// latency of detokenization operations in seconds
DECLARE_COUNTER(detokenization_latency_seconds_stream);
DECLARE_COUNTER(detokenization_latency_seconds_non_stream);

DECLARE_HISTOGRAM(prefix_cache_block_matched_rate);
DECLARE_HISTOGRAM(prefix_cache_block_matched_num);

// total number of model execution operations
DECLARE_COUNTER(num_model_execution_total_eager);

// latency of worker execution operations in seconds
DECLARE_COUNTER(execution_latency_seconds_model);
DECLARE_COUNTER(execution_latency_seconds_logits_processing);
DECLARE_COUNTER(execution_latency_seconds_sampling);

DECLARE_GAUGE(num_pending_requests);
DECLARE_GAUGE(num_running_requests);
DECLARE_GAUGE(num_waiting_requests);
DECLARE_GAUGE(num_preempted_requests);
DECLARE_GAUGE(num_offline_decode_preempt_offline_requests);
DECLARE_GAUGE(num_online_decode_preempt_online_requests);
DECLARE_GAUGE(num_online_prefill_preempt_offline_requests);
DECLARE_GAUGE(num_online_decode_preempt_offline_requests);
DECLARE_GAUGE(num_running_sequences);
DECLARE_GAUGE(kv_cache_utilization_perc);
DECLARE_GAUGE(num_blocks_in_prefix_cache);
DECLARE_GAUGE(num_free_blocks);
DECLARE_GAUGE(num_used_blocks);
DECLARE_COUNTER(scheduling_latency_seconds);

// total number of processing tokens
DECLARE_COUNTER(num_processing_tokens_total_prompt);
DECLARE_COUNTER(num_processing_tokens_total_generated);

DECLARE_HISTOGRAM(num_prompt_tokens_per_request);
DECLARE_HISTOGRAM(num_generated_tokens_per_request);

DECLARE_HISTOGRAM(time_to_first_token_latency_milliseconds);
DECLARE_HISTOGRAM(inter_token_latency_milliseconds);

// latency of responding in seconds
DECLARE_COUNTER(responsing_latency_seconds_stream);
DECLARE_COUNTER(responsing_latency_seconds_non_stream);

DECLARE_HISTOGRAM(end_2_end_latency_milliseconds);

// total number of request that server processed
DECLARE_COUNTER(server_request_in_total);
DECLARE_COUNTER(server_request_total_ok);
DECLARE_COUNTER(server_request_total_limit);
DECLARE_COUNTER(server_request_total_fail);

DECLARE_GAUGE(num_concurrent_requests);

DECLARE_GAUGE(xllm_cpu_num);
DECLARE_GAUGE(xllm_cpu_utilization);
DECLARE_GAUGE(xllm_gpu_num);
DECLARE_GAUGE(xllm_gpu_utilization);

// latency of speculative execution in seconds
DECLARE_COUNTER(speculative_execution_latency_seconds_draft);
DECLARE_COUNTER(speculative_execution_latency_seconds_target);
DECLARE_COUNTER(speculative_execution_latency_seconds_validation);
DECLARE_COUNTER(speculative_num_accepted_tokens_total);
DECLARE_COUNTER(speculative_num_draft_tokens_total);

// latency of proto conversion in seconds
DECLARE_COUNTER(proto_latency_seconds_proto2i);
DECLARE_COUNTER(proto_latency_seconds_i2proto);
DECLARE_COUNTER(proto_latency_seconds_proto2o);
DECLARE_COUNTER(proto_latency_seconds_o2proto);

// engine metrics
DECLARE_COUNTER(prepare_input_latency_seconds);

// rec engine metrics
DECLARE_COUNTER(prepare_input_latency_microseconds);
DECLARE_COUNTER(rec_first_token_latency_microseconds);
DECLARE_COUNTER(rec_second_token_latency_microseconds);
DECLARE_COUNTER(rec_third_token_latency_microseconds);
DECLARE_COUNTER(rec_sampling_latency_microseconds);
DECLARE_HISTOGRAM(expand_beam_latency_microseconds);

// multi node metrics
DECLARE_COUNTER(worker_service_latency_seconds);
DECLARE_COUNTER(engine_latency_seconds);

// PD disaggregation metrics
DECLARE_COUNTER_PER_MINUTE(disagg_pd_add_new_requests_total);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_add_new_requests_fail_total);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_add_new_requests_ok_total);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_add_new_requests_reject_total);
DECLARE_HISTOGRAM(disagg_pd_add_new_requests_latency_microseconds);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_first_generation_total);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_first_generation_fail_total);
DECLARE_HISTOGRAM(disagg_pd_first_generation_latency_microseconds);

// PD prefill queue (per-minute enqueue/dequeue)
DECLARE_COUNTER_PER_MINUTE(disagg_pd_prefill_queue_enqueue_total);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_prefill_queue_offline_enqueue_total);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_prefill_queue_dequeue_total);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_prefill_queue_offline_dequeue_total);

// PD decode: requests waiting for FirstGeneration from prefill
DECLARE_GAUGE(disagg_pd_received_request_map_size);

// PD: current number of linked P/D instances (P side: linked D instances)
DECLARE_GAUGE(disagg_pd_linked_instances_count);

// PD: per-instance failure distribution (dimension: instance name)
DECLARE_MULTI_COUNTER(disagg_pd_add_new_requests_fail_by_instance);
DECLARE_MULTI_COUNTER(disagg_pd_add_new_requests_reject_by_instance);
DECLARE_MULTI_COUNTER(disagg_pd_first_generation_fail_by_instance);

// PD PULL mode: KV cache pull latency (D side)
DECLARE_HISTOGRAM(disagg_pd_pull_kv_cache_latency_microseconds);

// LLM worker PUSH mode: push KV cache success/fail (per-minute)
DECLARE_COUNTER_PER_MINUTE(disagg_pd_push_kv_cache_ok_total);
DECLARE_COUNTER_PER_MINUTE(disagg_pd_push_kv_cache_fail_total);

// PUSH mode: collectAll wait latency (microseconds)
DECLARE_HISTOGRAM(disagg_pd_push_kv_cache_extra_wait_microseconds);

// memory metrics
DECLARE_GAUGE(total_memory_size_in_kilobytes);
DECLARE_GAUGE(weight_size_in_kilobytes);
DECLARE_GAUGE(total_kv_cache_size_in_kilobytes);
DECLARE_GAUGE(total_activation_size_in_kilobytes);

DECLARE_MULTI_HISTOGRAM(active_kv_cache_size_in_kilobytes);
DECLARE_MULTI_HISTOGRAM(prefill_active_activation_size_in_kilobytes);
DECLARE_MULTI_HISTOGRAM(decode_active_activation_size_in_kilobytes);
