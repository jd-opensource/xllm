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

#include <gtest/gtest.h>

#include <cstdint>

namespace xllm {
namespace {

BlockGroupSpec make_spec(int32_t group_id,
                         BlockGroupKind kind,
                         int32_t tokens_per_block,
                         int64_t num_blocks,
                         int32_t fixed_blocks_per_sequence,
                         bool enable_prefix_cache = false,
                         bool enable_cache_upload = false) {
  BlockGroupSpec spec;
  spec.group_id = group_id;
  spec.kind = kind;
  spec.tokens_per_block = tokens_per_block;
  spec.num_blocks = num_blocks;
  spec.fixed_blocks_per_sequence = fixed_blocks_per_sequence;
  spec.enable_prefix_cache = enable_prefix_cache;
  spec.enable_cache_upload = enable_cache_upload;
  return spec;
}

}  // namespace

TEST(BlockGroupPlanTest, AcceptsDenseTokenAndRingGroups) {
  CompositeBlockPlan plan;
  plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::RING,
                /*tokens_per_block=*/16,
                /*num_blocks=*/24,
                /*fixed_blocks_per_sequence=*/12),
      make_spec(/*group_id=*/1,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/64,
                /*num_blocks=*/128,
                /*fixed_blocks_per_sequence=*/0),
      make_spec(/*group_id=*/2,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/2048,
                /*num_blocks=*/32,
                /*fixed_blocks_per_sequence=*/0),
  };

  EXPECT_NO_FATAL_FAILURE(validate_composite_block_plan(plan));
}

TEST(BlockGroupPlanTest, RejectsEmptyPlan) {
  CompositeBlockPlan plan;

  EXPECT_DEATH(validate_composite_block_plan(plan), ".*");
}

TEST(BlockGroupPlanTest, RejectsNonDenseGroupIds) {
  CompositeBlockPlan plan;
  plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/64,
                /*num_blocks=*/128,
                /*fixed_blocks_per_sequence=*/0),
      make_spec(/*group_id=*/2,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/2048,
                /*num_blocks=*/32,
                /*fixed_blocks_per_sequence=*/0),
  };

  EXPECT_DEATH(validate_composite_block_plan(plan), ".*");
}

TEST(BlockGroupPlanTest, RejectsUnsortedGroupIds) {
  CompositeBlockPlan plan;
  plan.groups = {
      make_spec(/*group_id=*/1,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/64,
                /*num_blocks=*/128,
                /*fixed_blocks_per_sequence=*/0),
      make_spec(/*group_id=*/0,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/2048,
                /*num_blocks=*/32,
                /*fixed_blocks_per_sequence=*/0),
  };

  EXPECT_DEATH(validate_composite_block_plan(plan), ".*");
}

TEST(BlockGroupPlanTest, RejectsInvalidTokenGroups) {
  CompositeBlockPlan fixed_token_plan;
  fixed_token_plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/64,
                /*num_blocks=*/128,
                /*fixed_blocks_per_sequence=*/1),
  };

  EXPECT_DEATH(validate_composite_block_plan(fixed_token_plan), ".*");

  CompositeBlockPlan zero_tokens_plan;
  zero_tokens_plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/0,
                /*num_blocks=*/128,
                /*fixed_blocks_per_sequence=*/0),
  };

  EXPECT_DEATH(validate_composite_block_plan(zero_tokens_plan), ".*");
}

TEST(BlockGroupPlanTest, RejectsInvalidRingGroups) {
  CompositeBlockPlan variable_ring_plan;
  variable_ring_plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::RING,
                /*tokens_per_block=*/16,
                /*num_blocks=*/24,
                /*fixed_blocks_per_sequence=*/0),
  };

  EXPECT_DEATH(validate_composite_block_plan(variable_ring_plan), ".*");

  CompositeBlockPlan undersized_ring_plan;
  undersized_ring_plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::RING,
                /*tokens_per_block=*/16,
                /*num_blocks=*/8,
                /*fixed_blocks_per_sequence=*/12),
  };

  EXPECT_DEATH(validate_composite_block_plan(undersized_ring_plan), ".*");
}

TEST(BlockGroupPlanTest, RejectsPrefixCacheAndUpload) {
  CompositeBlockPlan prefix_plan;
  prefix_plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/64,
                /*num_blocks=*/128,
                /*fixed_blocks_per_sequence=*/0,
                /*enable_prefix_cache=*/true),
  };

  EXPECT_DEATH(validate_composite_block_plan(prefix_plan), ".*");

  CompositeBlockPlan upload_plan;
  upload_plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/64,
                /*num_blocks=*/128,
                /*fixed_blocks_per_sequence=*/0,
                /*enable_prefix_cache=*/false,
                /*enable_cache_upload=*/true),
  };

  EXPECT_DEATH(validate_composite_block_plan(upload_plan), ".*");
}

}  // namespace xllm
