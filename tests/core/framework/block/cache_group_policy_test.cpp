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

#include "framework/block/cache_group_policy.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "framework/block/block_manager_context.h"
#include "framework/block/block_manager_impl.h"
#include "framework/block/cache_group.h"

namespace xllm {
namespace {

constexpr uint32_t kBlockSize = 4;

std::unique_ptr<BlockManagerImpl> make_allocator(uint32_t num_blocks) {
  BlockManager::Options options;
  options.num_blocks(num_blocks)
      .block_size(kBlockSize)
      .enable_prefix_cache(false);
  return std::make_unique<BlockManagerImpl>(options);
}

CacheGroupSpec make_incremental_spec() {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::C1;
  spec.policy_type = CachePolicyType::INCREMENTAL_APPEND;
  spec.block_size = kBlockSize;
  return spec;
}

CacheGroupSpec make_rolling_spec(uint32_t window_blocks) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::SWA;
  spec.policy_type = CachePolicyType::ROLLING_WINDOW;
  spec.block_size = kBlockSize;
  spec.window_blocks = window_blocks;
  return spec;
}

CacheGroupSpec make_single_res_spec() {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::SINGLE_RES;
  spec.policy_type = CachePolicyType::PER_SEQUENCE_ONCE;
  spec.block_size = kBlockSize;
  return spec;
}

std::vector<int32_t> block_ids(const std::vector<Block>& blocks) {
  std::vector<int32_t> ids;
  ids.reserve(blocks.size());
  for (const Block& block : blocks) {
    ids.emplace_back(block.id());
  }
  return ids;
}

}  // namespace

TEST(CacheGroupTest, DefaultsAndToString) {
  CacheGroupSpec spec;
  EXPECT_EQ(spec.prefix_group, PrefixCacheGroup::INVALID);
  EXPECT_FALSE(spec.prefix_cacheable);
  EXPECT_EQ(spec.export_index, -1);
  EXPECT_EQ(spec.compress_ratio, 1u);

  CacheGroupState state;
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(state.next_logical_block_idx, 0u);
  EXPECT_EQ(state.ring_capacity, 0u);
  EXPECT_EQ(state.last_alloc_new_blocks, 0u);

  BlockManagerContext context;
  EXPECT_EQ(context.sequence, nullptr);
  EXPECT_EQ(context.kv_state, nullptr);
  EXPECT_EQ(context.role, CacheStorageRole::DEVICE);
  EXPECT_EQ(context.device_dp_rank, -1);

  EXPECT_STREQ(to_string(CachePolicyType::INCREMENTAL_APPEND),
               "incremental_append");
  EXPECT_STREQ(to_string(CachePolicyType::ROLLING_WINDOW), "rolling_window");
  EXPECT_STREQ(to_string(CachePolicyType::PER_SEQUENCE_ONCE),
               "per_sequence_once");
  EXPECT_STREQ(to_string(CacheStateId::SINGLE_RES), "single_res");
  EXPECT_STREQ(to_string(CacheStorageRole::HOST), "host");
  EXPECT_STREQ(to_string(WorkerExportTarget::MULTI_BLOCK_TABLES),
               "multi_block_tables");
}

TEST(CacheGroupPolicyFactoryTest, CreatesPolicyMatchingSpec) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/10);

  std::unique_ptr<ICacheGroupPolicy> incremental =
      create_cache_group_policy(make_incremental_spec(), allocator.get());
  EXPECT_NE(dynamic_cast<IncrementalAppendPolicy*>(incremental.get()), nullptr);

  std::unique_ptr<ICacheGroupPolicy> rolling = create_cache_group_policy(
      make_rolling_spec(/*window_blocks=*/4), allocator.get());
  EXPECT_NE(dynamic_cast<RollingWindowPolicy*>(rolling.get()), nullptr);

  std::unique_ptr<ICacheGroupPolicy> once =
      create_cache_group_policy(make_single_res_spec(), allocator.get());
  EXPECT_NE(dynamic_cast<PerSequenceOncePolicy*>(once.get()), nullptr);
}

TEST(IncrementalAppendPolicyTest, GrowsBlocksByTokenCeil) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/10);
  IncrementalAppendPolicy policy(make_incremental_spec(), allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/7));
  EXPECT_EQ(state.blocks.size(), 2u);
  EXPECT_EQ(state.last_alloc_new_blocks, 2u);

  // same token count maps to the same block count: no growth
  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/8));
  EXPECT_EQ(state.blocks.size(), 2u);
  EXPECT_EQ(state.last_alloc_new_blocks, 0u);

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/9));
  EXPECT_EQ(state.blocks.size(), 3u);
  EXPECT_EQ(state.last_alloc_new_blocks, 1u);
  EXPECT_EQ(allocator->num_free_blocks(), 6u);

  policy.deallocate(&context, &state);
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(allocator->num_free_blocks(), 9u);
  EXPECT_EQ(allocator->num_used_blocks(), 0u);
}

TEST(IncrementalAppendPolicyTest, FailedAllocateLeavesStateUnchanged) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/3);
  IncrementalAppendPolicy policy(make_incremental_spec(), allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  // 12 tokens need 3 blocks but only 2 are usable
  EXPECT_FALSE(policy.allocate(&context, &state, /*num_tokens=*/12));
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(allocator->num_free_blocks(), 2u);

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/8));
  EXPECT_EQ(state.blocks.size(), 2u);
  const std::vector<int32_t> ids_before = block_ids(state.blocks);

  EXPECT_FALSE(policy.allocate(&context, &state, /*num_tokens=*/12));
  EXPECT_EQ(block_ids(state.blocks), ids_before);

  policy.deallocate(&context, &state);
  EXPECT_EQ(allocator->num_free_blocks(), 2u);
}

TEST(IncrementalAppendPolicyTest, RollbackDropsOnlyLastAllocation) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/10);
  IncrementalAppendPolicy policy(make_incremental_spec(), allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/4));
  const std::vector<int32_t> ids_before = block_ids(state.blocks);

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/12));
  EXPECT_EQ(state.blocks.size(), 3u);

  policy.rollback(&context, &state);
  EXPECT_EQ(block_ids(state.blocks), ids_before);
  EXPECT_EQ(allocator->num_free_blocks(), 8u);

  // rollback is consumed: a second rollback must be a no-op
  policy.rollback(&context, &state);
  EXPECT_EQ(block_ids(state.blocks), ids_before);

  policy.deallocate(&context, &state);
}

TEST(PerSequenceOncePolicyTest, AllocatesOneBlockAndReusesIt) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/4);
  PerSequenceOncePolicy policy(make_single_res_spec(), allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/1));
  ASSERT_EQ(state.blocks.size(), 1u);
  const int32_t first_id = state.blocks[0].id();
  EXPECT_EQ(state.last_alloc_new_blocks, 1u);

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/100));
  ASSERT_EQ(state.blocks.size(), 1u);
  EXPECT_EQ(state.blocks[0].id(), first_id);
  EXPECT_EQ(state.last_alloc_new_blocks, 0u);
  EXPECT_EQ(allocator->num_free_blocks(), 2u);

  // last allocate added nothing, so rollback keeps the block
  policy.rollback(&context, &state);
  EXPECT_EQ(state.blocks.size(), 1u);

  policy.deallocate(&context, &state);
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(allocator->num_free_blocks(), 3u);
}

TEST(PerSequenceOncePolicyTest, RollbackReleasesFirstAllocation) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/4);
  PerSequenceOncePolicy policy(make_single_res_spec(), allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/1));
  policy.rollback(&context, &state);
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(allocator->num_free_blocks(), 3u);
}

TEST(PerSequenceOncePolicyTest, FailsWhenAllocatorExhausted) {
  // a single block pool is fully consumed by the reserved padding block
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/1);
  PerSequenceOncePolicy policy(make_single_res_spec(), allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  EXPECT_FALSE(policy.allocate(&context, &state, /*num_tokens=*/1));
  EXPECT_TRUE(state.blocks.empty());
}

TEST(RollingWindowPolicyTest, FirstAllocateOccupiesFullWindow) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/12);
  RollingWindowPolicy policy(make_rolling_spec(/*window_blocks=*/4),
                             allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  // even one token claims the full ring one-shot (admission rule)
  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/1));
  EXPECT_EQ(state.blocks.size(), 4u);
  EXPECT_EQ(state.ring_capacity, 4u);
  EXPECT_EQ(state.next_logical_block_idx, 1u);
  EXPECT_TRUE(state.pending_replacements.empty());
  EXPECT_EQ(allocator->num_free_blocks(), 7u);

  policy.deallocate(&context, &state);
  EXPECT_EQ(allocator->num_free_blocks(), 11u);
  EXPECT_EQ(state.ring_capacity, 0u);
}

TEST(RollingWindowPolicyTest, ReplacesRingSlotInPlaceBeyondWindow) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/12);
  RollingWindowPolicy policy(make_rolling_spec(/*window_blocks=*/4),
                             allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  // fill the window exactly: 16 tokens = 4 logical blocks, no replacement
  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/16));
  EXPECT_EQ(state.next_logical_block_idx, 4u);
  EXPECT_TRUE(state.pending_replacements.empty());
  const std::vector<int32_t> ring_ids = block_ids(state.blocks);

  // logical block 4 -> slot 0 replaced in place, other slots untouched
  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/18));
  EXPECT_EQ(state.blocks.size(), 4u);
  EXPECT_NE(state.blocks[0].id(), ring_ids[0]);
  EXPECT_EQ(state.blocks[1].id(), ring_ids[1]);
  EXPECT_EQ(state.blocks[2].id(), ring_ids[2]);
  EXPECT_EQ(state.blocks[3].id(), ring_ids[3]);
  ASSERT_EQ(state.pending_replacements.size(), 1u);
  EXPECT_EQ(state.pending_replacements[0].slot, 0u);
  EXPECT_EQ(state.pending_replacements[0].old_block.id(), ring_ids[0]);
  // the replaced-out block is retained, not yet returned to the allocator
  EXPECT_EQ(allocator->num_free_blocks(), 6u);

  // next allocate commits the pending replacement and frees the old block
  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/22));
  ASSERT_EQ(state.pending_replacements.size(), 1u);
  EXPECT_EQ(state.pending_replacements[0].slot, 1u);
  EXPECT_EQ(state.pending_replacements[0].old_block.id(), ring_ids[1]);
  EXPECT_EQ(allocator->num_free_blocks(), 6u);

  policy.deallocate(&context, &state);
  EXPECT_EQ(allocator->num_free_blocks(), 11u);
  EXPECT_EQ(allocator->num_used_blocks(), 0u);
}

TEST(RollingWindowPolicyTest, RollbackRestoresReplacedSlots) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/20);
  RollingWindowPolicy policy(make_rolling_spec(/*window_blocks=*/4),
                             allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/16));
  const std::vector<int32_t> ring_ids = block_ids(state.blocks);
  const size_t free_after_ring = allocator->num_free_blocks();

  // logical blocks 4..9 replace slots 0,1,2,3,0,1: slots 0/1 replaced twice
  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/40));
  EXPECT_EQ(state.pending_replacements.size(), 6u);
  EXPECT_EQ(state.next_logical_block_idx, 10u);

  policy.rollback(&context, &state);
  EXPECT_EQ(block_ids(state.blocks), ring_ids);
  EXPECT_EQ(state.next_logical_block_idx, 4u);
  EXPECT_TRUE(state.pending_replacements.empty());
  EXPECT_EQ(allocator->num_free_blocks(), free_after_ring);

  policy.deallocate(&context, &state);
}

TEST(RollingWindowPolicyTest, RollbackOfFirstAllocateClearsRing) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/10);
  RollingWindowPolicy policy(make_rolling_spec(/*window_blocks=*/4),
                             allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/4));
  policy.rollback(&context, &state);
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(state.ring_capacity, 0u);
  EXPECT_EQ(state.next_logical_block_idx, 0u);
  EXPECT_EQ(allocator->num_free_blocks(), 9u);
}

TEST(RollingWindowPolicyTest, FirstAllocateFailsWhenPoolSmallerThanWindow) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/4);
  RollingWindowPolicy policy(make_rolling_spec(/*window_blocks=*/4),
                             allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  // only 3 usable blocks, the ring needs 4 one-shot
  EXPECT_FALSE(policy.allocate(&context, &state, /*num_tokens=*/1));
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(state.ring_capacity, 0u);
  EXPECT_EQ(allocator->num_free_blocks(), 3u);
}

TEST(RollingWindowPolicyTest, MidReplacementFailureRestoresRing) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/5);
  RollingWindowPolicy policy(make_rolling_spec(/*window_blocks=*/4),
                             allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  EXPECT_TRUE(policy.allocate(&context, &state, /*num_tokens=*/16));
  const std::vector<int32_t> ring_ids = block_ids(state.blocks);
  EXPECT_EQ(allocator->num_free_blocks(), 0u);

  // replacement needs one new block but the pool is empty
  EXPECT_FALSE(policy.allocate(&context, &state, /*num_tokens=*/18));
  EXPECT_EQ(block_ids(state.blocks), ring_ids);
  EXPECT_EQ(state.ring_capacity, 4u);
  EXPECT_EQ(state.next_logical_block_idx, 4u);
  EXPECT_TRUE(state.pending_replacements.empty());

  policy.deallocate(&context, &state);
  EXPECT_EQ(allocator->num_free_blocks(), 4u);
}

TEST(RollingWindowPolicyTest, FirstAllocateWithReplacementFailureReleasesAll) {
  std::unique_ptr<BlockManagerImpl> allocator =
      make_allocator(/*num_blocks=*/5);
  RollingWindowPolicy policy(make_rolling_spec(/*window_blocks=*/4),
                             allocator.get());
  BlockManagerContext context;
  CacheGroupState state;

  // one call: first-shot ring (4 blocks) succeeds, then logical block 4
  // needs a 5th block that the pool cannot provide
  EXPECT_FALSE(policy.allocate(&context, &state, /*num_tokens=*/18));
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(state.ring_capacity, 0u);
  EXPECT_EQ(state.next_logical_block_idx, 0u);
  EXPECT_EQ(allocator->num_free_blocks(), 4u);
  EXPECT_EQ(allocator->num_used_blocks(), 0u);
}

}  // namespace xllm
