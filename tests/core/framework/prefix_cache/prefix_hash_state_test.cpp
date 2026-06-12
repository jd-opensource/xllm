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

#include "framework/prefix_cache/prefix_hash_state.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "framework/kv_cache/kv_cache_tensor_group.h"
#include "framework/prefix_cache/block_hasher.h"
#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {
namespace {

// Reference cumulative prefix key over [0, token_end) using the same primitive
// (xxh3_128bits_hash) that BlockHasher / PrefixCache::match / insert chain
// through, so a correct PrefixHashState key must be byte-identical to this.
XXH3Key reference_key(const std::vector<int32_t>& tokens,
                      uint32_t stride,
                      size_t token_end) {
  const Slice<int32_t> all(tokens);
  XXH3Key running;
  bool first = true;
  for (size_t i = 0; i < token_end; i += stride) {
    const uint8_t* pre = first ? nullptr : running.data;
    const Slice<int32_t> block =
        all.slice(static_cast<int32_t>(i), static_cast<int32_t>(i + stride));
    xxh3_128bits_hash(pre, block, running.data);
    first = false;
  }
  return running;
}

std::vector<int32_t> make_tokens(size_t n) {
  std::vector<int32_t> tokens;
  tokens.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    // Deterministic, non-trivial token stream.
    tokens.push_back(static_cast<int32_t>((i * 7 + 3) % 101));
  }
  return tokens;
}

constexpr uint32_t kStride = 4;

}  // namespace

TEST(PrefixHashStateTest, KeysMatchReferenceChaining) {
  const std::vector<int32_t> tokens = make_tokens(20);
  const Slice<int32_t> slice(tokens);

  PrefixHashState state;
  state.ensure(PrefixCacheGroup::C1, kStride, slice, tokens.size());

  for (size_t token_end = kStride; token_end <= tokens.size();
       token_end += kStride) {
    EXPECT_EQ(state.key(PrefixCacheGroup::C1, token_end).debug_string(),
              reference_key(tokens, kStride, token_end).debug_string())
        << "mismatch at token_end=" << token_end;
  }
}

TEST(PrefixHashStateTest, IncrementalEnsureEqualsOneShot) {
  const std::vector<int32_t> tokens = make_tokens(32);
  const Slice<int32_t> slice(tokens);

  PrefixHashState incremental;
  // Grow the chain in several steps; each ensure reuses the prior prefix.
  incremental.ensure(PrefixCacheGroup::C1, kStride, slice, 8);
  incremental.ensure(PrefixCacheGroup::C1, kStride, slice, 8);  // no-op
  incremental.ensure(PrefixCacheGroup::C1, kStride, slice, 20);
  incremental.ensure(PrefixCacheGroup::C1, kStride, slice, tokens.size());

  PrefixHashState one_shot;
  one_shot.ensure(PrefixCacheGroup::C1, kStride, slice, tokens.size());

  for (size_t token_end = kStride; token_end <= tokens.size();
       token_end += kStride) {
    EXPECT_EQ(incremental.key(PrefixCacheGroup::C1, token_end).debug_string(),
              one_shot.key(PrefixCacheGroup::C1, token_end).debug_string())
        << "mismatch at token_end=" << token_end;
  }
}

TEST(PrefixHashStateTest, PartialTrailingStrideIsNotCovered) {
  const std::vector<int32_t> tokens = make_tokens(10);  // 2 full strides + 2
  const Slice<int32_t> slice(tokens);

  PrefixHashState state;
  // Request more than the tokens back; only whole strides within tokens cover.
  state.ensure(PrefixCacheGroup::C1, kStride, slice, /*token_end=*/16);

  EXPECT_EQ(state.covered_tokens(PrefixCacheGroup::C1), 8u);
  EXPECT_TRUE(state.covers(PrefixCacheGroup::C1, 8));
  EXPECT_FALSE(state.covers(PrefixCacheGroup::C1, 12));
  // token_end == 0 is the empty prefix, always covered.
  EXPECT_TRUE(state.covers(PrefixCacheGroup::C1, 0));
}

TEST(PrefixHashStateTest, TruncateDropsKeysAtOrBeyondPosition) {
  const std::vector<int32_t> tokens = make_tokens(24);
  const Slice<int32_t> slice(tokens);

  PrefixHashState state;
  state.ensure(PrefixCacheGroup::C1, kStride, slice, tokens.size());
  ASSERT_EQ(state.covered_tokens(PrefixCacheGroup::C1), 24u);

  // Rewrite from token 9: keys covering token_end > 9 are invalid; the largest
  // surviving boundary is 8 (floor(9 / 4) * 4).
  state.truncate(9);
  EXPECT_EQ(state.covered_tokens(PrefixCacheGroup::C1), 8u);
  EXPECT_TRUE(state.covers(PrefixCacheGroup::C1, 8));
  EXPECT_FALSE(state.covers(PrefixCacheGroup::C1, 12));

  // Surviving keys are unchanged from the original chain.
  EXPECT_EQ(state.key(PrefixCacheGroup::C1, 8).debug_string(),
            reference_key(tokens, kStride, 8).debug_string());

  // Re-extending recomputes the same keys the original chain held.
  state.ensure(PrefixCacheGroup::C1, kStride, slice, tokens.size());
  EXPECT_EQ(state.key(PrefixCacheGroup::C1, 24).debug_string(),
            reference_key(tokens, kStride, 24).debug_string());
}

TEST(PrefixHashStateTest, GroupsWithDistinctStridesAreIndependent) {
  const std::vector<int32_t> tokens = make_tokens(32);
  const Slice<int32_t> slice(tokens);
  constexpr uint32_t kSingleStride = 8;

  PrefixHashState state;
  state.ensure(PrefixCacheGroup::C1, kStride, slice, tokens.size());
  state.ensure(PrefixCacheGroup::SINGLE, kSingleStride, slice, tokens.size());

  // Each group chains by its own stride and is byte-identical to its reference.
  EXPECT_EQ(state.key(PrefixCacheGroup::C1, 16).debug_string(),
            reference_key(tokens, kStride, 16).debug_string());
  EXPECT_EQ(state.key(PrefixCacheGroup::SINGLE, 16).debug_string(),
            reference_key(tokens, kSingleStride, 16).debug_string());

  // Different strides over the same tokens must produce different keys at the
  // same token_end.
  EXPECT_NE(state.key(PrefixCacheGroup::C1, 16).debug_string(),
            state.key(PrefixCacheGroup::SINGLE, 16).debug_string());

  state.reset();
  EXPECT_EQ(state.covered_tokens(PrefixCacheGroup::C1), 0u);
  EXPECT_EQ(state.covered_tokens(PrefixCacheGroup::SINGLE), 0u);
}

}  // namespace xllm
