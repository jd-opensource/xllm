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
#include "core/framework/config/config_json_utils.h"

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

DEFINE_int64(dit_generation_image_area_max,
             0,
             "Maximum allowed image area (width * height) for image generation "
             "requests. If set to 0, there is no limit.");

namespace xllm {

void DiTConfig::from_flags() {
  max_requests_per_batch(FLAGS_max_requests_per_batch)
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
      .dit_debug_print(FLAGS_dit_debug_print)
      .dit_generation_image_area_max(FLAGS_dit_generation_image_area_max);
}

void DiTConfig::from_json(const JsonReader& json) {
  max_requests_per_batch(json.value_or<int32_t>("max_requests_per_batch",
                                                max_requests_per_batch()))
      .dit_cache_policy(
          json.value_or<std::string>("dit_cache_policy", dit_cache_policy()))
      .dit_cache_warmup_steps(json.value_or<int64_t>("dit_cache_warmup_steps",
                                                     dit_cache_warmup_steps()))
      .dit_cache_n_derivatives(json.value_or<int64_t>(
          "dit_cache_n_derivatives", dit_cache_n_derivatives()))
      .dit_cache_skip_interval_steps(json.value_or<int64_t>(
          "dit_cache_skip_interval_steps", dit_cache_skip_interval_steps()))
      .dit_cache_residual_diff_threshold(
          json.value_or<double>("dit_cache_residual_diff_threshold",
                                dit_cache_residual_diff_threshold()))
      .dit_cache_start_steps(json.value_or<int64_t>("dit_cache_start_steps",
                                                    dit_cache_start_steps()))
      .dit_cache_end_steps(
          json.value_or<int64_t>("dit_cache_end_steps", dit_cache_end_steps()))
      .dit_cache_start_blocks(json.value_or<int64_t>("dit_cache_start_blocks",
                                                     dit_cache_start_blocks()))
      .dit_cache_end_blocks(json.value_or<int64_t>("dit_cache_end_blocks",
                                                   dit_cache_end_blocks()))
      .dit_sp_communication_overlap(json.value_or<int64_t>(
          "dit_sp_communication_overlap", dit_sp_communication_overlap()))
      .dit_debug_print(
          json.value_or<bool>("dit_debug_print", dit_debug_print()))
      .dit_generation_image_area_max(json.value_or<int64_t>(
          "dit_generation_image_area_max", dit_generation_image_area_max()));
}

DiTConfig& DiTConfig::get_instance() {
  static DiTConfig config;
  return config;
}

void DiTConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
