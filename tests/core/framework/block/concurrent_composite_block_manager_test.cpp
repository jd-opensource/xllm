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

#include "framework/block/concurrent_composite_block_manager.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <thread>
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

}  // namespace

TEST(ConcurrentCompositeBlockManagerTest, ForwardsSequenceLevelCalls) {
  ConcurrentCompositeBlockManager manager(
      {make_c1_spec(/*num_blocks=*/10),
       make_single_res_spec(/*num_blocks=*/4)});

  EXPECT_EQ(manager.num_groups(), 2u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 9u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SINGLE_RES), 3u);
  EXPECT_EQ(manager.num_free_blocks(), 12u);

  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;

  EXPECT_TRUE(manager.allocate(&context, /*num_tokens=*/9));
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 6u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SINGLE_RES), 2u);

  manager.deallocate(&context);
  EXPECT_TRUE(kv_state.groups().empty());
  EXPECT_EQ(manager.num_free_blocks(), 12u);
}

// The wrapper forwards flush/match to the orchestrator under its lock: sequence
// A flushes three committed C1 blocks, sequence B restores them via match.
TEST(ConcurrentCompositeBlockManagerTest, ForwardsPrefixCacheFlushAndMatch) {
  ConcurrentCompositeBlockManager manager(
      {make_cacheable_c1_spec(/*num_blocks=*/16),
       make_single_res_spec(/*num_blocks=*/4)});

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);

  KVCacheState kv_a;
  BlockManagerContext context_a;
  context_a.kv_state = &kv_a;
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/3 * kBlockSize));

  PrefixHashState hash_a;
  PrefixCacheInsertResult flushed =
      manager.flush_prefix_cache(&context_a,
                                 Slice<int32_t>(tokens),
                                 /*committed_tokens=*/3 * kBlockSize,
                                 &hash_a);
  EXPECT_EQ(flushed.inserted_blocks.size(), 3u);

  KVCacheState kv_b;
  BlockManagerContext context_b;
  context_b.kv_state = &kv_b;
  CompositeMatchResult matched =
      manager.match_prefix_cache(&context_b, Slice<int32_t>(tokens));

  EXPECT_EQ(matched.matched_tokens, 3u * kBlockSize);
  ASSERT_EQ(matched.group_matches.size(), 1u);
  EXPECT_EQ(matched.group_matches[0].blocks.size(), 3u);
}

// The wrapper forwards the pool-level accounting accessors to the orchestrator
// under its lock, so the BlockManagerPool sees the same totals it would from a
// monolithic manager.
TEST(ConcurrentCompositeBlockManagerTest, ForwardsBlockAccounting) {
  ConcurrentCompositeBlockManager manager(
      {make_cacheable_c1_spec(/*num_blocks=*/16),
       make_single_res_spec(/*num_blocks=*/4)});

  // Padding blocks excluded from total and uncounted as used; 15 + 3 usable,
  // nothing cached yet.
  EXPECT_EQ(manager.num_total_blocks(), 18u);
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(manager.kv_cache_utilization(), 0.0);

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);
  KVCacheState kv_state;
  BlockManagerContext context;
  context.kv_state = &kv_state;
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/3 * kBlockSize));
  EXPECT_EQ(manager.num_used_blocks(), 4u);

  PrefixHashState hash_state;
  manager.flush_prefix_cache(&context,
                             Slice<int32_t>(tokens),
                             /*committed_tokens=*/3 * kBlockSize,
                             &hash_state);
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 3u);
}

// Distinct sequences allocate/deallocate from separate threads while sharing
// the same leaf allocators. The bare BlockManagerImpl has a check-then-allocate
// window that aborts under unsynchronized concurrency; the wrapper's lock must
// close it, and the accounting must return to baseline once every thread frees
// its blocks.
TEST(ConcurrentCompositeBlockManagerTest,
     ConcurrentSequencesBalanceToBaseline) {
  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 200;
  // Generously sized so every thread can hold its peak (3 C1 + 1 SINGLE_RES)
  // simultaneously without exhausting either pool.
  ConcurrentCompositeBlockManager manager(
      {make_c1_spec(/*num_blocks=*/64),
       make_single_res_spec(/*num_blocks=*/32)});

  const size_t baseline_free = manager.num_free_blocks();
  ASSERT_EQ(baseline_free, 63u + 31u);

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t t = 0; t < kThreads; ++t) {
    workers.emplace_back([&manager]() {
      KVCacheState kv_state;
      BlockManagerContext context;
      context.kv_state = &kv_state;
      for (uint32_t i = 0; i < kIterations; ++i) {
        ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/4));
        ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/9));
        manager.deallocate(&context);
      }
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }

  EXPECT_EQ(manager.num_free_blocks(), baseline_free);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::C1), 63u);
  EXPECT_EQ(manager.group_free_blocks(CacheStateId::SINGLE_RES), 31u);
}

}  // namespace xllm
