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

#include "framework/block/composite_prefix_policy.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "framework/block/block_manager_impl.h"
#include "framework/block/cache_group.h"
#include "framework/prefix_cache/prefix_cache.h"
#include "framework/prefix_cache/prefix_hash_state.h"
#include "framework/request/sequence_kv_state.h"
#include "util/slice.h"

namespace xllm {
namespace {

constexpr uint32_t kStride = 4;

std::vector<int32_t> make_tokens(size_t n) {
  std::vector<int32_t> tokens;
  tokens.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    tokens.push_back(static_cast<int32_t>((i * 7 + 3) % 101));
  }
  return tokens;
}

// A leaf allocator with prefix caching disabled, mirroring the per-group
// allocator that GroupCompositeBlockManager builds for each cache group.
std::unique_ptr<BlockManagerImpl> make_leaf_allocator(uint32_t num_blocks) {
  BlockManager::Options options;
  options.num_blocks(num_blocks).block_size(kStride).enable_prefix_cache(false);
  return std::make_unique<BlockManagerImpl>(options);
}

// Pushes a C1 cache group holding `block_count` freshly-allocated blocks onto
// `kv_state`, returning the ids of those blocks in allocation order.
std::vector<int32_t> push_c1_group(KVCacheState* kv_state,
                                   BlockManagerImpl* allocator,
                                   size_t block_count) {
  std::vector<Block> blocks = allocator->allocate(block_count);
  std::vector<int32_t> ids;
  ids.reserve(blocks.size());
  for (const Block& block : blocks) {
    ids.push_back(block.id());
  }

  CacheGroupState state;
  state.state_id = CacheStateId::C1;
  state.blocks = std::move(blocks);
  kv_state->mutable_groups()->push_back(std::move(state));
  return ids;
}

CacheablePrefixEntry make_c1_entry(PrefixCache* cache) {
  CacheablePrefixEntry entry;
  entry.state_id = CacheStateId::C1;
  entry.group = PrefixCacheGroup::C1;
  entry.block_size = kStride;
  entry.cache = cache;
  return entry;
}

PrefixCacheInsertContext make_insert_context(KVCacheState* kv_state,
                                             const Slice<int32_t>& tokens,
                                             size_t committed_tokens,
                                             PrefixHashState* hash_state) {
  PrefixCacheInsertContext context;
  context.kv_state = kv_state;
  context.tokens = tokens;
  context.committed_tokens = committed_tokens;
  context.hash_state = hash_state;
  context.role = CacheStorageRole::DEVICE;
  context.device_dp_rank = 0;
  return context;
}

PrefixCacheMatchContext make_match_context(KVCacheState* kv_state,
                                           const Slice<int32_t>& tokens) {
  PrefixCacheMatchContext context;
  context.kv_state = kv_state;
  context.tokens = tokens;
  context.role = CacheStorageRole::DEVICE;
  context.device_dp_rank = 0;
  return context;
}

}  // namespace

// insert_committed stamps each newly-completed full C1 block with its
// cumulative prefix hash, inserts it into the cache, advances
// prefix_cached_tokens, and reports exactly the keys it had not seen before; a
// re-insert at the same boundary is a no-op.
TEST(CompositePrefixPolicyTest, IncrementalInsertIsIdempotent) {
  auto allocator = make_leaf_allocator(/*num_blocks=*/16);
  PrefixCache cache(kStride);
  IncrementalOnlyPrefixPolicy policy(make_c1_entry(&cache));

  const std::vector<int32_t> tokens = make_tokens(3 * kStride);
  KVCacheState kv_state;
  const std::vector<int32_t> block_ids =
      push_c1_group(&kv_state, allocator.get(), /*block_count=*/3);
  PrefixHashState hash_state;

  PrefixCacheInsertContext insert_ctx =
      make_insert_context(&kv_state,
                          Slice<int32_t>(tokens),
                          /*committed_tokens=*/3 * kStride,
                          &hash_state);
  PrefixCacheInsertResult inserted = policy.insert_committed(insert_ctx);

  EXPECT_EQ(inserted.inserted_blocks.size(), 3u);
  EXPECT_EQ(cache.num_blocks(), 3u);
  EXPECT_EQ(kv_state.group_state(CacheStateId::C1)->prefix_cached_tokens,
            3u * kStride);

  std::set<size_t> token_ends;
  for (const PrefixCacheInsertedBlock& block : inserted.inserted_blocks) {
    EXPECT_EQ(block.group, PrefixCacheGroup::C1);
    EXPECT_EQ(block.hash_stride, kStride);
    EXPECT_EQ(block.insert_kind, PrefixCacheInsertKind::FULL_BLOCK);
    EXPECT_EQ(block.role, CacheStorageRole::DEVICE);
    token_ends.insert(block.token_end);
  }
  EXPECT_EQ(token_ends, (std::set<size_t>{kStride, 2 * kStride, 3 * kStride}));

  // Re-inserting the same committed prefix inserts nothing new.
  PrefixCacheInsertResult again = policy.insert_committed(insert_ctx);
  EXPECT_TRUE(again.inserted_blocks.empty());
  EXPECT_EQ(cache.num_blocks(), 3u);
  (void)block_ids;
}

// A second sequence with the same prompt matches the cached prefix: match
// attaches the shared blocks to its C1 group and reports the restorable length.
TEST(CompositePrefixPolicyTest, MatchRestoresInsertedPrefix) {
  auto allocator = make_leaf_allocator(/*num_blocks=*/16);
  PrefixCache cache(kStride);
  IncrementalOnlyPrefixPolicy policy(make_c1_entry(&cache));

  const std::vector<int32_t> tokens = make_tokens(3 * kStride);

  // Sequence A inserts its 3 full blocks into the cache.
  KVCacheState kv_a;
  const std::vector<int32_t> a_ids =
      push_c1_group(&kv_a, allocator.get(), /*block_count=*/3);
  PrefixHashState hash_a;
  PrefixCacheInsertContext insert_ctx = make_insert_context(
      &kv_a, Slice<int32_t>(tokens), /*committed_tokens=*/3 * kStride, &hash_a);
  ASSERT_EQ(policy.insert_committed(insert_ctx).inserted_blocks.size(), 3u);

  // Sequence B holds an empty C1 group and matches the same prompt.
  KVCacheState kv_b;
  CacheGroupState empty_state;
  empty_state.state_id = CacheStateId::C1;
  kv_b.mutable_groups()->push_back(std::move(empty_state));

  PrefixCacheMatchContext match_ctx =
      make_match_context(&kv_b, Slice<int32_t>(tokens));
  CompositeMatchResult matched = policy.match(match_ctx);

  EXPECT_EQ(matched.matched_tokens, 3u * kStride);
  ASSERT_EQ(matched.group_matches.size(), 1u);
  EXPECT_EQ(matched.group_matches[0].group, PrefixCacheGroup::C1);
  EXPECT_EQ(matched.group_matches[0].blocks.size(), 3u);

  CacheGroupState* b_state = kv_b.group_state(CacheStateId::C1);
  ASSERT_NE(b_state, nullptr);
  EXPECT_EQ(b_state->shared_blocks_num, 3u);
  EXPECT_EQ(b_state->prefix_cached_tokens, 3u * kStride);
  ASSERT_EQ(b_state->blocks.size(), 3u);
  for (size_t i = 0; i < a_ids.size(); ++i) {
    EXPECT_EQ(b_state->blocks[i].id(), a_ids[i]);
  }
}

// match is only legal before the sequence holds any C1 block (first
// scheduling); the composite path has no re-match/replacement semantics, so a
// match against a populated group must CHECK-fail.
TEST(CompositePrefixPolicyTest, MatchDiesIfSequenceAlreadyHoldsBlocks) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  auto allocator = make_leaf_allocator(/*num_blocks=*/16);
  PrefixCache cache(kStride);
  IncrementalOnlyPrefixPolicy policy(make_c1_entry(&cache));

  const std::vector<int32_t> tokens = make_tokens(2 * kStride);
  KVCacheState kv_state;
  push_c1_group(&kv_state, allocator.get(), /*block_count=*/2);

  PrefixCacheMatchContext match_ctx =
      make_match_context(&kv_state, Slice<int32_t>(tokens));
  EXPECT_DEATH(policy.match(match_ctx),
               "prefix match must run before the sequence holds any block");
}

// insert_committed only commits whole blocks up to the committed-token
// boundary: with 10 committed tokens and stride 4, only the first two blocks
// (8 tokens) are inserted.
TEST(CompositePrefixPolicyTest, InsertStopsAtCommittedFullBlockBoundary) {
  auto allocator = make_leaf_allocator(/*num_blocks=*/16);
  PrefixCache cache(kStride);
  IncrementalOnlyPrefixPolicy policy(make_c1_entry(&cache));

  const std::vector<int32_t> tokens = make_tokens(3 * kStride);
  KVCacheState kv_state;
  push_c1_group(&kv_state, allocator.get(), /*block_count=*/3);
  PrefixHashState hash_state;

  PrefixCacheInsertContext insert_ctx = make_insert_context(
      &kv_state, Slice<int32_t>(tokens), /*committed_tokens=*/10, &hash_state);
  PrefixCacheInsertResult inserted = policy.insert_committed(insert_ctx);

  EXPECT_EQ(inserted.inserted_blocks.size(), 2u);
  EXPECT_EQ(cache.num_blocks(), 2u);
  EXPECT_EQ(kv_state.group_state(CacheStateId::C1)->prefix_cached_tokens,
            2u * kStride);
}

// NoPrefixPolicy never matches and never inserts, regardless of context.
TEST(CompositePrefixPolicyTest, NoPrefixPolicyReturnsEmpty) {
  NoPrefixPolicy policy;

  const std::vector<int32_t> tokens = make_tokens(3 * kStride);
  KVCacheState kv_state;
  PrefixHashState hash_state;

  PrefixCacheMatchContext match_ctx =
      make_match_context(&kv_state, Slice<int32_t>(tokens));
  CompositeMatchResult matched = policy.match(match_ctx);
  EXPECT_EQ(matched.matched_tokens, 0u);
  EXPECT_TRUE(matched.group_matches.empty());

  PrefixCacheInsertContext insert_ctx =
      make_insert_context(&kv_state,
                          Slice<int32_t>(tokens),
                          /*committed_tokens=*/3 * kStride,
                          &hash_state);
  PrefixCacheInsertResult inserted = policy.insert_committed(insert_ctx);
  EXPECT_TRUE(inserted.inserted_blocks.empty());
}

}  // namespace xllm
