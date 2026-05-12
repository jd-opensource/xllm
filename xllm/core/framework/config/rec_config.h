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

#pragma once

#include <cstdint>

#include "core/common/macros.h"

namespace xllm {

class RecConfig final {
 public:
  RecConfig() = default;
  ~RecConfig() = default;

  static RecConfig& get_instance();

  static RecConfig from_flags();
  static void reload_from_flags();

  PROPERTY(bool, enable_rec_fast_sampler) = true;

  PROPERTY(bool, enable_rec_prefill_only) = false;

  PROPERTY(bool, enable_xattention_one_stage) = false;

  PROPERTY(int32_t, max_decode_rounds) = 0;

  PROPERTY(bool, enable_constrained_decoding) = false;

  PROPERTY(bool, output_rec_logprobs) = false;

  PROPERTY(bool, enable_convert_tokens_to_item) = false;

  PROPERTY(bool, enable_output_sku_logprobs) = false;

  PROPERTY(bool, enable_extended_item_info) = false;

  PROPERTY(int32_t, each_conversion_threshold) = 50;

  PROPERTY(int32_t, total_conversion_threshold) = 1000;

  PROPERTY(int32_t, request_queue_size) = 100000;

  PROPERTY(uint32_t, rec_worker_max_concurrency) = 1;
};

}  // namespace xllm
