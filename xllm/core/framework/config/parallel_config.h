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
#include <string>

#include "core/common/macros.h"

namespace xllm {

class ParallelConfig final {
 public:
  ParallelConfig() = default;
  ~ParallelConfig() = default;

  static ParallelConfig& get_instance();

  static ParallelConfig from_flags();
  static void reload_from_flags();

  PROPERTY(int32_t, dp_size) = 1;

  PROPERTY(int32_t, ep_size) = 1;

  PROPERTY(int32_t, cp_size) = 1;

  PROPERTY(int64_t, tp_size) = 1;

  PROPERTY(int64_t, sp_size) = 1;

  PROPERTY(int64_t, cfg_size) = 1;

  PROPERTY(std::string, communication_backend) = "hccl";

  PROPERTY(bool, enable_prefill_sp) = false;

  PROPERTY(bool, enable_multi_stream_parallel) = false;

  PROPERTY(int32_t, micro_batch_num) = 1;

  PROPERTY(bool, enable_dp_balance) = false;
};

}  // namespace xllm
