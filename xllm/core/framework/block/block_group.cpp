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

#include "framework/block/block_group.h"

#include <glog/logging.h>

namespace xllm {

void validate_composite_block_plan(const CompositeBlockPlan& plan) {
  CHECK(!plan.groups.empty()) << "Composite block plan must not be empty";

  for (size_t i = 0; i < plan.groups.size(); ++i) {
    const BlockGroupSpec& group = plan.groups[i];
    CHECK_EQ(group.group_id, static_cast<int32_t>(i))
        << "Composite block group ids must be sorted and dense";
    CHECK_GT(group.tokens_per_block, 0) << "tokens_per_block must be positive";
    CHECK_GT(group.num_blocks, 0) << "num_blocks must be positive";
    CHECK(!group.enable_prefix_cache)
        << "Composite block groups do not support prefix cache";
    CHECK(!group.enable_cache_upload)
        << "Composite block groups do not support cache upload";

    switch (group.kind) {
      case BlockGroupKind::TOKEN:
        CHECK_EQ(group.fixed_blocks_per_sequence, 0)
            << "TOKEN block groups must not reserve fixed sequence blocks";
        break;
      case BlockGroupKind::RING:
        CHECK_GT(group.fixed_blocks_per_sequence, 0)
            << "RING block groups require fixed sequence blocks";
        CHECK_GE(group.num_blocks, group.fixed_blocks_per_sequence)
            << "RING block group capacity is smaller than one sequence";
        break;
      default:
        LOG(FATAL) << "Unknown block group kind";
    }
  }
}

}  // namespace xllm
