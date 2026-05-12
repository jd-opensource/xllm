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

#include "core/framework/config/load_config.h"

#include "core/common/global_flags.h"

DEFINE_bool(enable_manual_loader,
            false,
            "Pin decoder layer weights to host memory and use async H2D "
            "transfer. Required by enable_rolling_load; also implied by "
            "enable_xtensor.");

DEFINE_bool(enable_rolling_load,
            false,
            "Enable rolling weight load: keep only N decoder layer weight "
            "slots in HBM and stream-load each layer just-in-time. "
            "Requires enable_manual_loader=true. NPU only.");

DEFINE_int32(rolling_load_num_cached_layers,
             2,
             "Number of decoder layer weight slots to keep in HBM when "
             "enable_rolling_load=true.");

DEFINE_int32(rolling_load_num_rolling_slots,
             -1,
             "Number of rolling slots used by decoder rolling load. "
             "Fixed slots are computed as "
             "rolling_load_num_cached_layers - rolling_load_num_rolling_slots."
             " -1 means auto (min(2, preload_count)). "
             "Must be in [-1, rolling_load_num_cached_layers].");

DEFINE_bool(
    enable_prefetch_weight,
    false,
    "Whether to enable prefetch weight,only applicable to Qwen3-dense model."
    "The default prefetching ratio for gateup weight is 40%."
    "If adjustments are needed, e.g. export PREFETCH_COEFFOCIENT=0.5");

namespace xllm {

LoadConfig LoadConfig::from_flags() {
  LoadConfig config;
  config.enable_manual_loader(FLAGS_enable_manual_loader)
      .enable_rolling_load(FLAGS_enable_rolling_load)
      .rolling_load_num_cached_layers(FLAGS_rolling_load_num_cached_layers)
      .rolling_load_num_rolling_slots(FLAGS_rolling_load_num_rolling_slots)
      .enable_prefetch_weight(FLAGS_enable_prefetch_weight);
  return config;
}

LoadConfig& LoadConfig::get_instance() {
  static LoadConfig config = LoadConfig::from_flags();
  return config;
}

void LoadConfig::reload_from_flags() {
  LoadConfig::get_instance() = LoadConfig::from_flags();
}

}  // namespace xllm
