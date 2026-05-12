/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/config/scheduler_config.h"

#include "core/common/global_flags.h"

DEFINE_int32(max_tokens_per_batch, 10240, "Max number of tokens per batch.");

DEFINE_int32(max_seqs_per_batch, 1024, "Max number of sequences per batch.");

DEFINE_bool(enable_schedule_overlap,
            false,
            "Whether to enable schedule overlap.");

DEFINE_double(prefill_scheduling_memory_usage_threshold,
              0.95,
              "The memory usage threshold during prefill scheduling.");

DEFINE_bool(enable_chunked_prefill, true, "Whether to enable chunked prefill.");

DEFINE_int32(max_tokens_per_chunk_for_prefill,
             -1,
             "Max number of token per chunk in prefill stage.");

DEFINE_int32(chunked_match_frequency,
             2,
             "Number of sequence prefix cache match frequency.");

DEFINE_bool(use_zero_evict,
            false,
            "Use ZeroEvictionScheduler but ContinuousScheduler.");

DEFINE_int32(
    max_decode_token_per_sequence,
    256,
    "Max decode token per sequence which used for ZeroEvictionScheduler.");

DEFINE_string(priority_strategy,
              "fcfs",
              "Priority strategy for requests(e.g. fcfs, priority, deadline).");

DEFINE_bool(use_mix_scheduler,
            false,
            "Use MixScheduler for handling prefill and decode uniformly.");

DEFINE_bool(enable_online_preempt_offline,
            true,
            "Whether to enable online preempt offline.");

DEFINE_double(aggressive_coeff,
              1.0,
              "Aggressive coefficient for MixScheduler urgency judgment.");

DEFINE_double(starve_threshold,
              1.0,
              "Starvation threshold coefficient for MixScheduler.");

DEFINE_bool(enable_starve_prevent,
            true,
            "Whether to enable anti-starvation in MixScheduler.");

namespace xllm {

SchedulerConfig SchedulerConfig::from_flags() {
  SchedulerConfig config;
  config.max_tokens_per_batch(FLAGS_max_tokens_per_batch)
      .max_seqs_per_batch(FLAGS_max_seqs_per_batch)
      .enable_schedule_overlap(FLAGS_enable_schedule_overlap)
      .prefill_scheduling_memory_usage_threshold(
          FLAGS_prefill_scheduling_memory_usage_threshold)
      .enable_chunked_prefill(FLAGS_enable_chunked_prefill)
      .max_tokens_per_chunk_for_prefill(FLAGS_max_tokens_per_chunk_for_prefill)
      .chunked_match_frequency(FLAGS_chunked_match_frequency)
      .use_zero_evict(FLAGS_use_zero_evict)
      .max_decode_token_per_sequence(FLAGS_max_decode_token_per_sequence)
      .priority_strategy(FLAGS_priority_strategy)
      .use_mix_scheduler(FLAGS_use_mix_scheduler)
      .enable_online_preempt_offline(FLAGS_enable_online_preempt_offline)
      .aggressive_coeff(FLAGS_aggressive_coeff)
      .starve_threshold(FLAGS_starve_threshold)
      .enable_starve_prevent(FLAGS_enable_starve_prevent);
  return config;
}

SchedulerConfig& SchedulerConfig::get_instance() {
  static SchedulerConfig config = SchedulerConfig::from_flags();
  return config;
}

void SchedulerConfig::reload_from_flags() {
  SchedulerConfig::get_instance() = SchedulerConfig::from_flags();
}

}  // namespace xllm
