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
#include <vector>

#include "framework/block/cache_group.h"

namespace xllm {

// Model-agnostic inputs for laying out a model's cache groups. The engine owns
// capacity planning and supplies the per-group allocator sizing here; this
// builder owns only the structural mapping fixed by the design doc (which
// states exist, their policies, prefix groups, block sizes, prefix-cacheable
// flags, and worker export order). Exactly one model shape is produced:
//   - DSV4         when swa_window_blocks > 0 or compress_ratios is non-empty
//   - Qwen3.5+     when has_linear_state is set (C1 present but non-cacheable)
//   - normal       otherwise (C1 prefix-cacheable)
// SINGLE_RES is appended last for every shape.
struct ModelCacheGroupConfig {
  // Base (C1 / SWA) block size in tokens; compressed groups derive from it.
  uint32_t base_block_size = 0;

  // C1 self-attention allocator pool size. Used by the normal / Qwen shapes.
  uint32_t c1_num_blocks = 0;

  // Universal per-sequence resource group (SINGLE_RES); always emitted.
  uint32_t single_res_num_blocks = 0;

  // DSV4 sliding-window group. swa_window_blocks == 0 omits the SWA group.
  uint32_t swa_window_blocks = 0;
  uint32_t swa_num_blocks = 0;

  // DSV4 compressed groups: index-aligned ratio / allocator sizing. Empty for
  // non-DSV4 shapes. Phase-1 supported ratios: 4 (C4) and 128 (C128).
  std::vector<uint32_t> compress_ratios;
  std::vector<uint32_t> compress_num_blocks;

  // Qwen3.5+ hybrid (linear-attention) model. Forces every group non-cacheable:
  // a C1 prefix hit that skips prefill would strand the unrecoverable linear
  // recurrent state, corrupting output.
  bool has_linear_state = false;
};

// Build the ordered cache-group specs for a model in worker export order.
// CHECK-fails on an inconsistent config (e.g. an unsupported compress ratio, a
// ratio/sizing length mismatch, or a linear-state model with a cacheable
// group).
std::vector<CacheGroupSpec> build_cache_group_specs(
    const ModelCacheGroupConfig& config);

}  // namespace xllm
