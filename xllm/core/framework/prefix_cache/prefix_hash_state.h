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

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "block_hasher.h"
#include "core/framework/multimodal/mm_data.h"
#include "framework/kv_cache/kv_cache_tensor_group.h"
#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {

// Incremental cumulative-prefix-hash cache shared by a sequence across forward
// passes. For each PrefixCacheGroup (one hash domain per group), it stores the
// chained XXH3 keys at every stride boundary so that flush-time prefix-cache
// inserts and match-time lookups never recompute the hash from token 0.
//
//   key_k (token_end = (k + 1) * stride)
//       = hash(key_{k-1}, tokens[k * stride : (k + 1) * stride])
//
// with key_{-1} treated as the empty pre-hash. This reproduces the chaining in
// PrefixCache::match / PrefixCache::insert exactly, so a key produced here is
// byte-identical to the hash those paths write onto a Block.
//
// A single state object serves every group: C1/C4/C128 chain by their own
// block size, SINGLE chains by the restore chunk size. Only the stride differs.
class PrefixHashState final {
 public:
  explicit PrefixHashState(BlockHasherType hasher_type = BlockHasherType::TEXT);

  // Extends `group`'s chain so it covers `token_end`, reusing whatever prefix
  // is already computed. `stride` is the group's hash_stride and must stay
  // constant for a given group across calls. Only whole strides within `tokens`
  // are hashed; a trailing partial stride and any request beyond `tokens` are
  // left uncovered. `mm_data` is consumed only by the MM hasher (VLM) and
  // ignored by the TEXT hasher, mirroring PrefixCache::match / insert.
  void ensure(PrefixCacheGroup group,
              uint32_t stride,
              const Slice<int32_t>& tokens,
              size_t token_end,
              const MMData& mm_data = MMData());

  // Cumulative key at `token_end`, which must be a covered multiple of stride.
  const XXH3Key& key(PrefixCacheGroup group, size_t token_end) const;

  // Whether `group`'s chain already covers `token_end`.
  bool covers(PrefixCacheGroup group, size_t token_end) const;

  // Token position currently covered by `group`'s chain (num keys * stride).
  size_t covered_tokens(PrefixCacheGroup group) const;

  // Drops every key whose covered range reaches at or beyond `token_pos`, i.e.
  // keeps only keys with token_end <= token_pos. Used when tokens are rewritten
  // (beam-search rewrite, preemption reset, schedule-overlap fake-token fixup).
  void truncate(size_t token_pos);

  // Forgets all cached keys for every group.
  void reset();

 private:
  struct GroupChain {
    uint32_t stride = 0;
    // keys[k] is the cumulative key at token_end = (k + 1) * stride.
    std::vector<XXH3Key> keys;
  };

  static size_t group_index(PrefixCacheGroup group);

  GroupChain& chain_for(PrefixCacheGroup group, uint32_t stride);

  BlockHasherType hasher_type_;
  // Indexed by PrefixCacheGroup value: C1=0, SINGLE=1, C4=2, C128=3.
  std::array<GroupChain, 4> chains_;
};

}  // namespace xllm
