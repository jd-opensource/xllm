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

class ExecutionConfig final {
 public:
  ExecutionConfig() = default;
  ~ExecutionConfig() = default;

  static ExecutionConfig& get_instance();

  static ExecutionConfig from_flags();
  static void reload_from_flags();

  PROPERTY(bool, enable_graph) = false;

  PROPERTY(bool, enable_graph_mode_decode_no_padding) = false;

  PROPERTY(bool, enable_prefill_piecewise_graph) = false;

  PROPERTY(bool, enable_graph_vmm_pool) = true;

  PROPERTY(int32_t, max_tokens_for_graph_mode) = 2048;

  PROPERTY(bool, enable_shm) = false;

  PROPERTY(bool, use_contiguous_input_buffer) = true;

  PROPERTY(uint64_t, input_shm_size) = 1024;

  PROPERTY(uint64_t, output_shm_size) = 128;

  PROPERTY(int32_t, random_seed) = -1;

  PROPERTY(bool, enable_customize_mla_kernel) = false;

#if defined(USE_NPU)
  PROPERTY(std::string, npu_kernel_backend) = "AUTO";

  PROPERTY(bool, enable_intralayer_addnorm) = false;
#endif
};

}  // namespace xllm
