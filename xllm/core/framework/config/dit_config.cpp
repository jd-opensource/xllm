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

#include "core/framework/config/dit_config.h"

#include "core/common/global_flags.h"

DEFINE_int32(max_requests_per_batch, 1, "Max number of request per batch.");

DEFINE_string(dit_cache_policy,
              "TaylorSeer",
              "The policy of dit cache(e.g. None, FBCache, TaylorSeer, "
              "FBCacheTaylorSeer, ResidualCache).");

DEFINE_int64(dit_cache_warmup_steps, 0, "The number of warmup steps.");

DEFINE_int64(dit_cache_n_derivatives,
             3,
             "The number of derivatives to use in TaylorSeer.");

DEFINE_int64(dit_cache_skip_interval_steps,
             3,
             "The interval steps to skip for derivative calculation.");

DEFINE_double(dit_cache_residual_diff_threshold,
              0.09,
              "The residual difference threshold for cache reuse.");

DEFINE_int64(dit_cache_start_steps,
             5,
             "The number of steps to skip at the start");

DEFINE_int64(dit_cache_end_steps, 5, "The number of steps to skip at the end.");

DEFINE_int64(dit_cache_start_blocks,
             5,
             "The number of blocks to skip at the start.");

DEFINE_int64(dit_cache_end_blocks,
             5,
             "The number of blocks to skip at the end.");

DEFINE_int64(dit_sp_communication_overlap,
             1,
             "Communication & Computation overlap for sequence parallel");

DEFINE_bool(dit_debug_print,
            false,
            "whether print the debug info for dit models");

namespace xllm {

DiTConfig DiTConfig::from_flags() {
  DiTConfig config;
  config.max_requests_per_batch(FLAGS_max_requests_per_batch)
      .dit_cache_policy(FLAGS_dit_cache_policy)
      .dit_cache_warmup_steps(FLAGS_dit_cache_warmup_steps)
      .dit_cache_n_derivatives(FLAGS_dit_cache_n_derivatives)
      .dit_cache_skip_interval_steps(FLAGS_dit_cache_skip_interval_steps)
      .dit_cache_residual_diff_threshold(
          FLAGS_dit_cache_residual_diff_threshold)
      .dit_cache_start_steps(FLAGS_dit_cache_start_steps)
      .dit_cache_end_steps(FLAGS_dit_cache_end_steps)
      .dit_cache_start_blocks(FLAGS_dit_cache_start_blocks)
      .dit_cache_end_blocks(FLAGS_dit_cache_end_blocks)
      .dit_sp_communication_overlap(FLAGS_dit_sp_communication_overlap)
      .dit_debug_print(FLAGS_dit_debug_print);
  return config;
}

DiTConfig& DiTConfig::get_instance() {
  static DiTConfig config = DiTConfig::from_flags();
  return config;
}

void DiTConfig::reload_from_flags() {
  DiTConfig::get_instance() = DiTConfig::from_flags();
}

}  // namespace xllm
