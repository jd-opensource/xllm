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

#include "framework/block/group_composite_block_manager.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "framework/block/block_manager_context.h"
#include "framework/block/cache_group.h"
#include "framework/prefix_cache/prefix_hash_state.h"
#include "framework/request/sequence_kv_state.h"
#include "util/slice.h"

namespace xllm {
namespace {

constexpr uint32_t kBlockSize = 4;

CacheGroupSpec make_c1_spec(uint32_t num_blocks) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::C1;
  spec.policy_type = CachePolicyType::INCREMENTAL_APPEND;
  spec.block_size = kBlockSize;
  spec.num_blocks = num_blocks;
  return spec;
}

CacheGroupSpec make_cacheable_c1_spec(uint32_t num_blocks) {
  CacheGroupSpec spec = make_c1_spec(num_blocks);
  spec.prefix_cacheable = true;
  spec.prefix_group = PrefixCacheGroup::C1;
  return spec;
}

std::vector<int32_t> make_tokens(size_t n) {
  std::vector<int32_t> tokens;
  tokens.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    tokens.push_back(static_cast<int32_t>((i * 7 + 3) % 101));
  }
  return tokens;
}

CacheGroupSpec make_single_res_spec(uint32_t num_blocks) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::SINGLE_RES;
  spec.policy_type = CachePolicyType::PER_SEQUENCE_ONCE;
  spec.block_size = kBlockSize;
  spec.num_blocks = num_blocks;
  return spec;
}

CacheGroupSpec make_swa_spec(uint32_t window_blocks, uint32_t num_blocks) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::SWA;
  spec.policy_type = CachePolicyType::ROLLING_WINDOW;
  spec.block_size = kBlockSize;
  spec.window_blocks = window_blocks;
  spec.num_blocks = num_blocks;
  return spec;
}

size_t group_block_count(KVCacheState* kv_state, CacheStateId state_id) {
  CacheGroupState* state = kv_state->group_state(state_id);
  return state == nullptr ? 0 : state->blocks.size();
}

}  // namespace

TEST(GroupCompositeBlockManagerTest, ConstructsGroupsFromSpecs) {
  GroupCompositeBlockManager manager({make_c1_spec(/*num_blocks=*/10),
                                      make_single_res_spec(/*num_blocks=*/4)});

  EXPECT_EQ(manager.num_groups(), 2u);
  // each allocator reserves block 0 for padding, so free == num_blocks - 1
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 9u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SINGLE_RES), 3u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C128), 0u);
  EXPECT_EQ(manager.num_free_blocks(), 12u);
}

TEST(GroupCompositeBlockManagerTest, AllocateGrowsEveryGroup) {
  GroupCompositeBlockManager manager({make_c1_spec(/*num_blocks=*/10),
                                      make_single_res_spec(/*num_blocks=*/4)});

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;

  // 9 tokens -> ceil(9/4) = 3 C1 blocks; SINGLE_RES claims one block.
  EXPECT_TRUE(manager.allocate(&context, /*num_tokens=*/9));
  ASSERT_EQ(kv_state.groups().size(), 2u);
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::C1), 3u);
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::SINGLE_RES), 1u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 6u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SINGLE_RES), 2u);

  // same token count needs no further C1 growth; SINGLE_RES reuses its block.
  EXPECT_TRUE(manager.allocate(&context, /*num_tokens=*/10));
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::C1), 3u);
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::SINGLE_RES), 1u);

  // crossing the next block boundary grows only C1.
  EXPECT_TRUE(manager.allocate(&context, /*num_tokens=*/13));
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::C1), 4u);
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::SINGLE_RES), 1u);
}

TEST(GroupCompositeBlockManagerTest, AllocateRollsBackEarlierGroupsOnFailure) {
  // group order matters: C1 succeeds first, then SINGLE_RES (whose single-block
  // pool is fully consumed by its padding) fails, forcing C1 to roll back.
  GroupCompositeBlockManager manager({make_c1_spec(/*num_blocks=*/10),
                                      make_single_res_spec(/*num_blocks=*/1)});

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;

  EXPECT_FALSE(manager.allocate(&context, /*num_tokens=*/9));
  // C1 grew then unwound; SINGLE_RES never held a block.
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::C1), 0u);
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::SINGLE_RES), 0u);
  // every allocator is back to its post-construction free count.
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 9u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SINGLE_RES), 0u);
}

TEST(GroupCompositeBlockManagerTest,
     PartialFailureLeavesAllocatorsRecoverable) {
  GroupCompositeBlockManager manager({make_c1_spec(/*num_blocks=*/10),
                                      make_single_res_spec(/*num_blocks=*/1)});

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;

  EXPECT_FALSE(manager.allocate(&context, /*num_tokens=*/9));
  // after the failed attempt the group states persist but hold no blocks, so a
  // later request that only touches the healthy group must still be servable.
  ASSERT_EQ(kv_state.groups().size(), 2u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 9u);
}

TEST(GroupCompositeBlockManagerTest, DeallocateReleasesAllGroups) {
  GroupCompositeBlockManager manager({make_c1_spec(/*num_blocks=*/10),
                                      make_swa_spec(/*window_blocks=*/4,
                                                    /*num_blocks=*/12)});

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;

  EXPECT_TRUE(manager.allocate(&context, /*num_tokens=*/9));
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::C1), 3u);
  // SWA first allocate claims the full 4-block ring one-shot.
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::SWA), 4u);
  EXPECT_LT(manager.num_free_blocks(), 9u + 11u);

  manager.deallocate(&context);
  EXPECT_TRUE(kv_state.groups().empty());
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 9u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SWA), 11u);
  EXPECT_EQ(manager.num_free_blocks(), 20u);

  // deallocate is idempotent once the groups are gone.
  manager.deallocate(&context);
  EXPECT_TRUE(kv_state.groups().empty());
}

TEST(GroupCompositeBlockManagerTest, ReallocatesAfterDeallocate) {
  GroupCompositeBlockManager manager({make_c1_spec(/*num_blocks=*/10),
                                      make_single_res_spec(/*num_blocks=*/4)});

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;

  EXPECT_TRUE(manager.allocate(&context, /*num_tokens=*/8));
  manager.deallocate(&context);
  ASSERT_TRUE(kv_state.groups().empty());

  // a fresh allocation rebuilds the per-group state from scratch.
  EXPECT_TRUE(manager.allocate(&context, /*num_tokens=*/5));
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::C1), 2u);
  EXPECT_EQ(group_block_count(&kv_state, CacheStateId::SINGLE_RES), 1u);
}

// A prefix-cacheable C1 group flushes its committed blocks into a group-local
// cache; a second sequence with the same prompt restores them via match.
TEST(GroupCompositeBlockManagerTest, PrefixCacheRoundtripAcrossSequences) {
  GroupCompositeBlockManager manager({make_cacheable_c1_spec(/*num_blocks=*/16),
                                      make_single_res_spec(/*num_blocks=*/4)});

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);

  // Sequence A grows three C1 blocks and flushes them.
  KVCacheState kv_a;
  BlockManagerContext context_a;
  context_a.kv_state = &kv_a;
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/3 * kBlockSize));
  ASSERT_EQ(group_block_count(&kv_a, CacheStateId::C1), 3u);

  PrefixHashState hash_a;
  PrefixCacheInsertResult flushed =
      manager.flush_prefix_cache(&context_a,
                                 Slice<int32_t>(tokens),
                                 /*committed_tokens=*/3 * kBlockSize,
                                 &hash_a);
  EXPECT_EQ(flushed.inserted_blocks.size(), 3u);
  EXPECT_EQ(kv_a.group_state(CacheStateId::C1)->prefix_cached_tokens,
            3u * kBlockSize);

  // Sequence B matches the same prompt and restores the three shared blocks.
  KVCacheState kv_b;
  BlockManagerContext context_b;
  context_b.kv_state = &kv_b;
  CompositeMatchResult matched =
      manager.match_prefix_cache(&context_b, Slice<int32_t>(tokens));

  EXPECT_EQ(matched.matched_tokens, 3u * kBlockSize);
  ASSERT_EQ(matched.group_matches.size(), 1u);
  EXPECT_EQ(matched.group_matches[0].group, PrefixCacheGroup::C1);
  EXPECT_EQ(matched.group_matches[0].blocks.size(), 3u);

  CacheGroupState* b_state = kv_b.group_state(CacheStateId::C1);
  ASSERT_NE(b_state, nullptr);
  EXPECT_EQ(b_state->shared_blocks_num, 3u);
  EXPECT_EQ(b_state->prefix_cached_tokens, 3u * kBlockSize);
}

// num_used / num_total / kv_cache_utilization aggregate across every group
// allocator; num_blocks_in_prefix_cache is recovered from the C1 prefix cache
// because the leaf pools run with prefix caching disabled.
TEST(GroupCompositeBlockManagerTest, AggregatesBlockAccountingAcrossGroups) {
  GroupCompositeBlockManager manager({make_cacheable_c1_spec(/*num_blocks=*/16),
                                      make_single_res_spec(/*num_blocks=*/4)});

  // Post-construction each leaf reserves block 0 as padding (excluded from
  // total and not counted as used): total = 15 + 3, used = 0.
  EXPECT_EQ(manager.num_total_blocks(), 18u);
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(manager.kv_cache_utilization(), 0.0);

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;
  // 3 C1 blocks + 1 SINGLE_RES block are charged to used.
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/3 * kBlockSize));
  EXPECT_EQ(manager.num_used_blocks(), 4u);
  EXPECT_EQ(manager.num_total_blocks(), 18u);
  EXPECT_NEAR(manager.kv_cache_utilization(), 4.0 / 18.0, 1e-9);

  // Flushing the three committed C1 blocks populates the group-local prefix
  // cache; the non-cacheable SINGLE_RES group contributes nothing.
  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);
  PrefixHashState hash_state;
  manager.flush_prefix_cache(&context,
                             Slice<int32_t>(tokens),
                             /*committed_tokens=*/3 * kBlockSize,
                             &hash_state);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 3u);
}

// When every leaf holds only its padding block there is no usable capacity, so
// kv_cache_utilization returns 0 instead of dividing by a zero total.
TEST(GroupCompositeBlockManagerTest, UtilizationIsZeroWhenNoCapacity) {
  GroupCompositeBlockManager manager(
      {make_c1_spec(/*num_blocks=*/1), make_single_res_spec(/*num_blocks=*/1)});

  EXPECT_EQ(manager.num_total_blocks(), 0u);
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.kv_cache_utilization(), 0.0);
}

// With no prefix-cacheable group, the composite installs the NoPrefix policy:
// match restores nothing and flush inserts nothing.
TEST(GroupCompositeBlockManagerTest, NonCacheableGroupsDisablePrefixCache) {
  GroupCompositeBlockManager manager({make_c1_spec(/*num_blocks=*/16),
                                      make_single_res_spec(/*num_blocks=*/4)});

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/3 * kBlockSize));

  PrefixHashState hash_state;
  PrefixCacheInsertResult flushed =
      manager.flush_prefix_cache(&context,
                                 Slice<int32_t>(tokens),
                                 /*committed_tokens=*/3 * kBlockSize,
                                 &hash_state);
  EXPECT_TRUE(flushed.inserted_blocks.empty());

  CompositeMatchResult matched =
      manager.match_prefix_cache(&context, Slice<int32_t>(tokens));
  EXPECT_EQ(matched.matched_tokens, 0u);
  EXPECT_TRUE(matched.group_matches.empty());
}

}  // namespace xllm
