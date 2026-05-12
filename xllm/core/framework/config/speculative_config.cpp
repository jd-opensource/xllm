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

#include "core/framework/config/speculative_config.h"

#include "core/common/global_flags.h"

DEFINE_string(draft_model, "", "draft hf model path to the model file.");

DEFINE_string(draft_devices,
              "npu:0",
              "Devices to run the draft model on, e.g. npu:0, npu:0,npu:1.");

DEFINE_int32(num_speculative_tokens, 0, "Number of speculative tokens.");

DEFINE_string(speculative_algorithm,
              "MTP",
              "Speculative decoding algorithm. Supported options: MTP, Eagle3, "
              "Suffix. Default is MTP.");

DEFINE_int32(speculative_suffix_cache_max_depth,
             64,
             "Maximum suffix tree depth for suffix speculative decoding.");

DEFINE_double(speculative_suffix_max_spec_factor,
              1.0,
              "Suffix speculation max tokens factor relative to match length.");

DEFINE_double(speculative_suffix_max_spec_offset,
              0.0,
              "Suffix speculation max tokens additive offset.");

DEFINE_double(speculative_suffix_min_token_prob,
              0.1,
              "Minimum token probability used in suffix speculation.");

DEFINE_int32(speculative_suffix_max_cached_requests,
             -1,
             "Maximum globally cached requests for suffix speculation (-1 "
             "unlimited, 0 disabled).");

DEFINE_bool(speculative_suffix_use_tree_spec,
            false,
            "Whether to use tree-based suffix speculation instead of path "
            "speculation.");

DEFINE_bool(enable_opt_validate_probs,
            false,
            "Whether validate uses selected-only draft_probs [B,S] directly. "
            "If false, selected-only cache values are restored to dense "
            "[B,S,V].");

DEFINE_bool(enable_atb_spec_kernel,
            false,
            "Whether to use ATB speculative kernel.");

namespace xllm {

SpeculativeConfig SpeculativeConfig::from_flags() {
  SpeculativeConfig config;
  config.draft_model(FLAGS_draft_model)
      .draft_devices(FLAGS_draft_devices)
      .num_speculative_tokens(FLAGS_num_speculative_tokens)
      .speculative_algorithm(FLAGS_speculative_algorithm)
      .speculative_suffix_cache_max_depth(
          FLAGS_speculative_suffix_cache_max_depth)
      .speculative_suffix_max_spec_factor(
          FLAGS_speculative_suffix_max_spec_factor)
      .speculative_suffix_max_spec_offset(
          FLAGS_speculative_suffix_max_spec_offset)
      .speculative_suffix_min_token_prob(
          FLAGS_speculative_suffix_min_token_prob)
      .speculative_suffix_max_cached_requests(
          FLAGS_speculative_suffix_max_cached_requests)
      .speculative_suffix_use_tree_spec(FLAGS_speculative_suffix_use_tree_spec)
      .enable_opt_validate_probs(FLAGS_enable_opt_validate_probs)
      .enable_atb_spec_kernel(FLAGS_enable_atb_spec_kernel);
  return config;
}

SpeculativeConfig& SpeculativeConfig::get_instance() {
  static SpeculativeConfig config = SpeculativeConfig::from_flags();
  return config;
}

void SpeculativeConfig::reload_from_flags() {
  SpeculativeConfig::get_instance() = SpeculativeConfig::from_flags();
}

}  // namespace xllm
