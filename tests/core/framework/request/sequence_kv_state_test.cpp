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
  BlockManagerContext context_a;
  context_a.kv_state = &kv_a;
  ASSERT_TRUE(manager.allocate(&context_a, /*num_tokens=*/3 * kBlockSize));
  PrefixHashState hash_a;
  manager.flush_prefix_cache(&context_a,
                             Slice<int32_t>(tokens),
                             /*committed_tokens=*/3 * kBlockSize,
                             &hash_a);

  KVCacheState kv_b;
  BlockManagerContext context_b;
  context_b.kv_state = &kv_b;
  manager.match_prefix_cache(&context_b, Slice<int32_t>(tokens));

  EXPECT_EQ(kv_b.shared_kv_blocks_num(), 3u);
  EXPECT_EQ(kv_b.shared_kv_tokens_num(), 3u * kBlockSize);
}

}  // namespace xllm
