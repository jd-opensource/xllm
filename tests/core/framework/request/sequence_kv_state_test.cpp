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

#include "framework/request/sequence_kv_state.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "framework/block/block_manager_context.h"
#include "framework/block/cache_group.h"
#include "framework/block/group_composite_block_manager.h"
#include "framework/prefix_cache/prefix_hash_state.h"
#include "util/slice.h"

namespace xllm {
namespace {

constexpr uint32_t kBlockSize = 4;

CacheGroupSpec make_cacheable_c1_spec(uint32_t num_blocks) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::C1;
  spec.policy_type = CachePolicyType::INCREMENTAL_APPEND;
  spec.block_size = kBlockSize;
  spec.num_blocks = num_blocks;
  spec.prefix_cacheable = true;
  spec.prefix_group = PrefixCacheGroup::C1;
  return spec;
}

CacheGroupSpec make_single_res_spec(uint32_t num_blocks) {
  CacheGroupSpec spec;
  spec.state_id = CacheStateId::SINGLE_RES;
  spec.policy_type = CachePolicyType::PER_SEQUENCE_ONCE;
  spec.block_size = kBlockSize;
  spec.num_blocks = num_blocks;
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

}  // namespace

TEST(KVCacheStateTest, TransferCursorTracksAndResets) {
  KVCacheState state;
  EXPECT_EQ(state.next_transfer_block_idx(), 0u);

  state.set_next_transfer_block_idx(2);
  EXPECT_EQ(state.next_transfer_block_idx(), 2u);

  state.advance_transfer_block_idx(5);
  EXPECT_EQ(state.next_transfer_block_idx(), 5u);

  state.advance_transfer_block_idx(3);
  EXPECT_EQ(state.next_transfer_block_idx(), 5u);

  state.reset();
  EXPECT_EQ(state.next_transfer_block_idx(), 0u);
}

// With no per-group state, every legacy view stays anchored on the flat
// blocks_ list -- the monolithic path must be byte-for-byte unchanged.
TEST(KVCacheStateTest, LegacyFlatViewsUnchangedWhenNoGroups) {
  KVCacheState state;
  EXPECT_FALSE(state.on_composite_path());
  EXPECT_TRUE(state.kv_blocks().empty());
  EXPECT_EQ(state.num_kv_blocks(), 0u);
  EXPECT_EQ(state.shared_kv_blocks_num(), 0u);
  EXPECT_EQ(state.shared_kv_tokens_num(), 0u);
  EXPECT_EQ(state.current_max_tokens_capacity(), 0u);
}

// On the composite path the C1 group backs kv_blocks() / num_kv_blocks() /
// capacity, so the worker block-table path keeps reading the attention blocks.
TEST(KVCacheStateTest, C1GroupBacksLegacyViewsOnCompositePath) {
  GroupCompositeBlockManager manager({make_cacheable_c1_spec(/*num_blocks=*/16),
                                      make_single_res_spec(/*num_blocks=*/4)});

  KVCacheState state;
  BlockManagerContext context;
  context.kv_state = &state;
  ASSERT_TRUE(manager.allocate(&context, /*num_tokens=*/3 * kBlockSize));

  EXPECT_TRUE(state.on_composite_path());
  // C1 holds three blocks; SINGLE_RES is invisible to the flat view.
  EXPECT_EQ(state.num_kv_blocks(), 3u);
  EXPECT_EQ(state.kv_blocks().size(), 3u);
  Slice<Block> c1_blocks = state.group_blocks(CacheStateId::C1);
  ASSERT_EQ(c1_blocks.size(), 3u);
  EXPECT_EQ(state.kv_blocks()[0].id(), c1_blocks[0].id());
  EXPECT_EQ(state.kv_blocks()[2].id(), c1_blocks[2].id());
  // Only the incremental C1 group contributes a linear token capacity.
  EXPECT_EQ(state.current_max_tokens_capacity(), 3u * kBlockSize);
}

// Shared-block accounting reads through to the C1 group after a prefix match,
// so the scheduler sees the restored prefix length.
TEST(KVCacheStateTest, SharedCountsReadThroughC1GroupAfterMatch) {
  GroupCompositeBlockManager manager({make_cacheable_c1_spec(/*num_blocks=*/16),
                                      make_single_res_spec(/*num_blocks=*/4)});

  const std::vector<int32_t> tokens = make_tokens(3 * kBlockSize);

  KVCacheState kv_a;
  PrefixHashState hash_a;
  BlockManagerContext context_a;
  context_a.kv_state = &kv_a;
  context_a.tokens = Slice<int32_t>(tokens);
  context_a.hash_state = &hash_a;
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/3 * kBlockSize));
  // Commit the three blocks, then the next allocate lazily inserts them into
  // the prefix cache.
  kv_a.set_kv_cache_tokens_num(3 * kBlockSize);
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/3 * kBlockSize));

  KVCacheState kv_b;
  BlockManagerContext context_b;
  context_b.kv_state = &kv_b;
  manager.match_prefix_cache(&context_b, Slice<int32_t>(tokens));

  EXPECT_EQ(kv_b.shared_kv_blocks_num(), 3u);
  EXPECT_EQ(kv_b.shared_kv_tokens_num(), 3u * kBlockSize);
}

// multi_block_table_groups() returns only the groups that export to the
// worker's multi_block_tables (export_index >= 0), ordered by export_index, so
// DSV4 assembles its SWA/C4/C128 tables in worker slot order regardless of the
// groups_ vector order. The non-exported SINGLE_RES group (export_index < 0) is
// skipped.
TEST(KVCacheStateTest, MultiBlockTableGroupsOrderedByExportIndex) {
  KVCacheState state;
  std::vector<CacheGroupState>* groups = state.mutable_groups();

  // Deliberately push the groups out of export order.
  CacheGroupState single;
  single.state_id = CacheStateId::SINGLE_RES;
  single.export_index = -1;
  groups->push_back(single);

  CacheGroupState c128;
  c128.state_id = CacheStateId::C128;
  c128.export_index = 2;
  groups->push_back(c128);

  CacheGroupState swa;
  swa.state_id = CacheStateId::SWA;
  swa.export_index = 0;
  groups->push_back(swa);

  CacheGroupState c4;
  c4.state_id = CacheStateId::C4;
  c4.export_index = 1;
  groups->push_back(c4);

  std::vector<const CacheGroupState*> exported =
      state.multi_block_table_groups();
  ASSERT_EQ(exported.size(), 3u);
  EXPECT_EQ(exported[0]->state_id, CacheStateId::SWA);
  EXPECT_EQ(exported[1]->state_id, CacheStateId::C4);
  EXPECT_EQ(exported[2]->state_id, CacheStateId::C128);
}

// The normal/Qwen flat path carries a single C1 group whose export_index is the
// -1 sentinel (the worker reads C1 through the flat block-table path, not
// multi_block_tables), so no group is exported.
TEST(KVCacheStateTest, MultiBlockTableGroupsEmptyForFlatC1) {
  KVCacheState state;
  CacheGroupState c1;
  c1.state_id = CacheStateId::C1;
  c1.export_index = -1;
  state.mutable_groups()->push_back(c1);

  EXPECT_TRUE(state.multi_block_table_groups().empty());
}

namespace {

// Build a CacheGroupState holding `count` blocks each `block_size` tokens wide.
CacheGroupState make_group_with_blocks(CacheStateId state_id,
                                       uint32_t block_size,
                                       size_t count) {
  CacheGroupState group;
  group.state_id = state_id;
  for (size_t i = 0; i < count; ++i) {
    group.blocks.emplace_back(block_size);
  }
  return group;
}

}  // namespace

// DSV4 composite state has no C1 group, so current_max_tokens_capacity() is
// min over the compressed incremental groups (C4/C128). The SWA ring's
// windowed capacity is excluded -- it would falsely cap token growth.
TEST(KVCacheStateTest, CurrentMaxTokensCapacityReadsC4GroupForDsv4) {
  KVCacheState state;
  std::vector<CacheGroupState>* groups = state.mutable_groups();
  // SWA ring of 5 narrow blocks: a window, not a linear capacity.
  groups->push_back(
      make_group_with_blocks(CacheStateId::SWA, /*block_size=*/kBlockSize, 5));
  // C4 group: 3 blocks of kBlockSize*4 = 12*kBlockSize tokens.
  groups->push_back(make_group_with_blocks(CacheStateId::C4,
                                           /*block_size=*/kBlockSize * 4, 3));
  // C128 group: 2 blocks of kBlockSize*128 = 256*kBlockSize tokens -> the
  // finer C4 group is the binding min.
  groups->push_back(make_group_with_blocks(CacheStateId::C128,
                                           /*block_size=*/kBlockSize * 128, 2));

  EXPECT_TRUE(state.on_composite_path());
  EXPECT_EQ(state.current_max_tokens_capacity(), 3u * kBlockSize * 4u);
}

// min(C4, C128) is a true min: when the C128 group's token capacity is the
// smaller one, it bounds the budget even though C4 is present.
TEST(KVCacheStateTest, CurrentMaxTokensCapacityTakesMinOfCompressedGroups) {
  KVCacheState state;
  std::vector<CacheGroupState>* groups = state.mutable_groups();
  // C4: 64 blocks * (kBlockSize*4) = 256*kBlockSize tokens.
  groups->push_back(make_group_with_blocks(CacheStateId::C4,
                                           /*block_size=*/kBlockSize * 4, 64));
  // C128: 1 block * (kBlockSize*128) = 128*kBlockSize tokens -> the min.
  groups->push_back(make_group_with_blocks(CacheStateId::C128,
                                           /*block_size=*/kBlockSize * 128, 1));

  EXPECT_EQ(state.current_max_tokens_capacity(), 1u * kBlockSize * 128u);
}

// When a DSV4 sequence carries C128 but no C4 group, the min ranges over the
// only compressed group present, never the SWA ring.
TEST(KVCacheStateTest, CurrentMaxTokensCapacityFallsBackToC128WhenNoC4) {
  KVCacheState state;
  std::vector<CacheGroupState>* groups = state.mutable_groups();
  groups->push_back(
      make_group_with_blocks(CacheStateId::SWA, /*block_size=*/kBlockSize, 5));
  groups->push_back(make_group_with_blocks(CacheStateId::C128,
                                           /*block_size=*/kBlockSize * 128, 2));

  EXPECT_EQ(state.current_max_tokens_capacity(), 2u * kBlockSize * 128u);
}

// A composite DSV4 state whose compressed groups hold no blocks yet contributes
// no linear capacity (the SWA ring never stands in for it).
TEST(KVCacheStateTest, CurrentMaxTokensCapacityZeroWhenCompressedGroupsEmpty) {
  KVCacheState state;
  std::vector<CacheGroupState>* groups = state.mutable_groups();
  groups->push_back(
      make_group_with_blocks(CacheStateId::SWA, /*block_size=*/kBlockSize, 5));
  groups->push_back(
      make_group_with_blocks(CacheStateId::C4, /*block_size=*/kBlockSize * 4, 0));

  EXPECT_TRUE(state.on_composite_path());
  EXPECT_EQ(state.current_max_tokens_capacity(), 0u);
}

}  // namespace xllm
