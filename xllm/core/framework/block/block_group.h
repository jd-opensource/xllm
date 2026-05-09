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

namespace xllm {

enum class BlockGroupKind : int32_t {
  TOKEN = 0,
  RING = 1,
};

struct BlockGroupSpec {
  int32_t group_id = 0;
  BlockGroupKind kind = BlockGroupKind::TOKEN;
  int32_t tokens_per_block = 0;
  int64_t num_blocks = 0;
  int32_t fixed_blocks_per_sequence = 0;
  bool enable_prefix_cache = false;
  bool enable_cache_upload = false;
};

struct CompositeBlockPlan {
  std::vector<BlockGroupSpec> groups;
};

void validate_composite_block_plan(const CompositeBlockPlan& plan);

}  // namespace xllm
