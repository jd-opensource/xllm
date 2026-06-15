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
  // Schedulable capacity is the C1 pool alone; SINGLE_RES never joins it.
  EXPECT_EQ(manager.num_free_blocks(), 9u);
}

// Admission capacity follows the simplified per-shape rule: with a C1 group it
// is exactly the C1 free count; without one (DSV4) it is the min over the
// compressed incremental groups in base-block equivalents. SWA and SINGLE_RES
// never contribute.
TEST(GroupCompositeBlockManagerTest, NumFreeBlocksUsesCompressedMinForDsv4) {
  CacheGroupSpec c4 = make_c1_spec(/*num_blocks=*/9);
  c4.state_id = CacheStateId::C4;
  c4.compress_ratio = 4;
  CacheGroupSpec c128 = make_c1_spec(/*num_blocks=*/2);
  c128.state_id = CacheStateId::C128;
  c128.compress_ratio = 128;

  GroupCompositeBlockManager manager(
      {make_swa_spec(/*window_blocks=*/4, /*num_blocks=*/12),
       c4,
       c128,
       make_single_res_spec(/*num_blocks=*/4)});

  // C4: 8 free * 4 = 32; C128: 1 free * 128 = 128 -> min is the C4 budget.
  EXPECT_EQ(manager.num_free_blocks(), 32u);
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
  // Schedulable capacity tracks the C1 pool only (SWA is windowed, not
  // token-linear).
  EXPECT_EQ(manager.num_free_blocks(), 6u);

  manager.deallocate(&context);
  EXPECT_TRUE(kv_state.groups().empty());
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 9u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SWA), 11u);
  EXPECT_EQ(manager.num_free_blocks(), 9u);

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

// A prefix-cacheable C1 group inserts its committed blocks into a group-local
// cache from inside allocate (lazy flush); a second sequence with the same
// prompt restores them via match.
TEST(GroupCompositeBlockManagerTest, PrefixCacheRoundtripAcrossSequences) {
  GroupCompositeBlockManager manager({make_cacheable_c1_spec(/*num_blocks=*/16),
                                      make_single_res_spec(/*num_blocks=*/4)});

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);

  // Sequence A grows three C1 blocks; the context carries the token view and
  // hash chain so the composite can insert completed blocks internally.
  KVCacheState kv_a;
  PrefixHashState hash_a;
  BlockManagerContext context_a;
  context_a.kv_state = &kv_a;
  context_a.tokens = Slice<int32_t>(tokens);
  context_a.hash_state = &hash_a;
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/3 * kBlockSize));
  ASSERT_EQ(group_block_count(&kv_a, CacheStateId::C1), 3u);

  // Once the three blocks commit, the next allocate lazily inserts them into
  // the group-local prefix cache before (re)growing.
  kv_a.set_kv_cache_tokens_num(3 * kBlockSize);
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/3 * kBlockSize));
  EXPECT_EQ(kv_a.group_state(CacheStateId::C1)->prefix_cached_tokens,
            3u * kBlockSize);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 3u);

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

// num_used / num_total / kv_cache_utilization describe the schedulable KV pool
// only -- the same C1 (normal) / compressed (DSV4) groups that num_free_blocks
// reports, never the per-sequence SINGLE_RES resource. This keeps the trio
// self-consistent and matches the scheduler, which multiplies num_used_blocks
// by the KV block_size and compares its sum against the request's KV block
// count (continuous_scheduler.cpp): a SINGLE_RES block (its own block_size,
// often 1) folded into that count would corrupt both. num_blocks_in_prefix_cache
// is recovered from the C1 prefix cache because the leaf pools run with prefix
// caching disabled.
TEST(GroupCompositeBlockManagerTest, AccountsOnlyTheSchedulableKvPool) {
  GroupCompositeBlockManager manager({make_cacheable_c1_spec(/*num_blocks=*/16),
                                      make_single_res_spec(/*num_blocks=*/4)});

  // The C1 leaf reserves block 0 as padding (excluded from total, never used),
  // so total = 15 and used = 0. The SINGLE_RES pool is outside this count.
  EXPECT_EQ(manager.num_total_blocks(), 15u);
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(manager.kv_cache_utilization(), 0.0);

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);
  KVCacheState kv_state;
  PrefixHashState hash_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;
  context.tokens = Slice<int32_t>(tokens);
  context.hash_state = &hash_state;
  // Only the 3 C1 blocks count as used; the SINGLE_RES block this sequence also
  // claims is a per-sequence resource, not part of the schedulable KV pool.
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/3 * kBlockSize));
  EXPECT_EQ(manager.num_used_blocks(), 3u);
  EXPECT_EQ(manager.num_total_blocks(), 15u);
  EXPECT_NEAR(manager.kv_cache_utilization(), 3.0 / 15.0, 1e-9);

  // Once committed, the next allocate lazily inserts the three C1 blocks into
  // the group-local prefix cache; the non-cacheable SINGLE_RES group
  // contributes nothing.
  kv_state.set_kv_cache_tokens_num(3 * kBlockSize);
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/3 * kBlockSize));
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
// the internal insert adds nothing and match restores nothing.
TEST(GroupCompositeBlockManagerTest, NonCacheableGroupsDisablePrefixCache) {
  GroupCompositeBlockManager manager({make_c1_spec(/*num_blocks=*/16),
                                      make_single_res_spec(/*num_blocks=*/4)});

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);

  KVCacheState kv_state;
  PrefixHashState hash_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;
  context.tokens = Slice<int32_t>(tokens);
  context.hash_state = &hash_state;
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/3 * kBlockSize));

  // Even after the blocks commit, the NoPrefix policy inserts nothing on the
  // next allocate's lazy flush.
  kv_state.set_kv_cache_tokens_num(3 * kBlockSize);
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/3 * kBlockSize));
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);

  CompositeMatchResult matched =
      manager.match_prefix_cache(&context, Slice<int32_t>(tokens));
  EXPECT_EQ(matched.matched_tokens, 0u);
  EXPECT_TRUE(matched.group_matches.empty());
}

// Group-local allocation fallback: when the C1 leaf pool is exhausted but its
// prefix cache still pins reclaimable blocks, allocate() evicts exactly the
// shortfall from that group's cache and the retry succeeds.
TEST(GroupCompositeBlockManagerTest, AllocateEvictsGroupPrefixCacheOnShortage) {
  // Free baseline is 3 (block 0 is padding) -- room for one 3-block sequence.
  GroupCompositeBlockManager manager(
      {make_cacheable_c1_spec(/*num_blocks=*/4)});

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);

  // Sequence A fills the pool then, on deallocate, the internal tail flush
  // inserts its three committed blocks into the C1 cache: the live blocks are
  // released but the cache still pins all three physical ids out of the free
  // list.
  KVCacheState kv_a;
  PrefixHashState hash_a;
  BlockManagerContext context_a;
  context_a.kv_state = &kv_a;
  context_a.tokens = Slice<int32_t>(tokens);
  context_a.hash_state = &hash_a;
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/3 * kBlockSize));
  kv_a.set_kv_cache_tokens_num(3 * kBlockSize);
  manager.deallocate(&context_a);
  ASSERT_EQ(manager.num_blocks_in_prefix_cache(), 3u);
  ASSERT_EQ(manager.group_free_blocks(CacheStateId::C1), 0u);

  // Sequence B needs three fresh blocks but the free list is empty. Without the
  // fallback the leaf returns nothing; with it the group evicts its cache and
  // the retry allocates the reclaimed ids. A bare context (no tokens/hash)
  // performs no insert of its own.
  KVCacheState kv_b;
  BlockManagerContext context_b;
  context_b.kv_state = &kv_b;
  EXPECT_TRUE(manager.allocate(&context_b, /*num_tokens=*/3 * kBlockSize));
  EXPECT_EQ(group_block_count(&kv_b, CacheStateId::C1), 3u);
  // The cache gave up exactly the blocks B reclaimed.
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
}

// When even a full eviction of the failing group's cache cannot cover the
// request, allocate() fails, rolls back the groups already grown in this call,
// and leaves the eviction in place (the cache is not part of the reservation).
TEST(GroupCompositeBlockManagerTest,
     AllocateEvictionFallbackStillFailsWhenCacheTooSmall) {
  // SINGLE_RES is ordered first so it grows successfully and must be rolled
  // back when the cacheable C1 group fails after exhausting its cache.
  GroupCompositeBlockManager manager(
      {make_single_res_spec(/*num_blocks=*/4),
       make_cacheable_c1_spec(/*num_blocks=*/4)});

  const std::vector<int32_t> tokens = make_tokens(2 * kBlockSize);

  // Sequence A caches two C1 blocks (8 tokens) via the deallocate tail flush,
  // leaving the C1 leaf with one free block and two pinned in the cache.
  KVCacheState kv_a;
  PrefixHashState hash_a;
  BlockManagerContext context_a;
  context_a.kv_state = &kv_a;
  context_a.tokens = Slice<int32_t>(tokens);
  context_a.hash_state = &hash_a;
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/2 * kBlockSize));
  kv_a.set_kv_cache_tokens_num(2 * kBlockSize);
  manager.deallocate(&context_a);
  ASSERT_EQ(manager.num_blocks_in_prefix_cache(), 2u);
  ASSERT_EQ(manager.group_free_blocks(CacheStateId::C1), 1u);

  // Sequence B needs four C1 blocks (16 tokens). Even after evicting both
  // cached blocks the leaf has only three free, so the allocate fails.
  KVCacheState kv_b;
  BlockManagerContext context_b;
  context_b.kv_state = &kv_b;
  EXPECT_FALSE(manager.allocate(&context_b, /*num_tokens=*/4 * kBlockSize));

  // C1 grew nothing; the eviction persisted (cache emptied, three ids free).
  EXPECT_EQ(group_block_count(&kv_b, CacheStateId::C1), 0u);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 3u);
  // SINGLE_RES grew its block first, then rolled back to its full free pool.
  EXPECT_EQ(group_block_count(&kv_b, CacheStateId::SINGLE_RES), 0u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SINGLE_RES), 3u);
}

// Each spec's export_index is copied onto its per-sequence group state when the
// composite materializes the states, so the worker can later emit the
// multi_block_tables slots in DSV4's SWA/C4/C128 order. SINGLE_RES keeps the
// sentinel -1 and is therefore excluded from multi_block_table_groups().
TEST(GroupCompositeBlockManagerTest, StampsExportIndexOntoGroupStates) {
  CacheGroupSpec swa = make_swa_spec(/*window_blocks=*/4, /*num_blocks=*/12);
  swa.export_index = 0;
  CacheGroupSpec c4 = make_c1_spec(/*num_blocks=*/16);
  c4.state_id = CacheStateId::C4;
  c4.export_index = 1;
  CacheGroupSpec single = make_single_res_spec(/*num_blocks=*/4);
  single.export_index = -1;

  GroupCompositeBlockManager manager({swa, c4, single});

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/9));

  ASSERT_EQ(kv_state.groups().size(), 3u);
  EXPECT_EQ(kv_state.group_state(CacheStateId::SWA)->export_index, 0);
  EXPECT_EQ(kv_state.group_state(CacheStateId::C4)->export_index, 1);
  EXPECT_EQ(kv_state.group_state(CacheStateId::SINGLE_RES)->export_index, -1);

  // The exported groups come back in ascending export_index order; the
  // SINGLE_RES group (export_index < 0) is skipped.
  std::vector<const CacheGroupState*> exported =
      kv_state.multi_block_table_groups();
  ASSERT_EQ(exported.size(), 2u);
  EXPECT_EQ(exported[0]->state_id, CacheStateId::SWA);
  EXPECT_EQ(exported[1]->state_id, CacheStateId::C4);
}

}  // namespace xllm
