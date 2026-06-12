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

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "core/framework/multimodal/mm_data.h"
#include "framework/prefix_cache/prefix_cache.h"
#include "framework/prefix_cache/prefix_hash_state.h"
#include "framework/request/sequence_kv_state.h"

namespace xllm {

CompositeMatchResult NoPrefixPolicy::match(
    const PrefixCacheMatchContext& /*context*/) {
  return CompositeMatchResult{};
}

PrefixCacheInsertResult NoPrefixPolicy::flush(
    const PrefixCacheFlushContext& /*context*/) {
  return PrefixCacheInsertResult{};
}

IncrementalOnlyPrefixPolicy::IncrementalOnlyPrefixPolicy(
    const CacheablePrefixEntry& c1_entry)
    : c1_(c1_entry) {
  CHECK(c1_.cache != nullptr)
      << "IncrementalOnlyPrefixPolicy requires a C1 prefix cache";
  CHECK_GT(c1_.block_size, 0u);
}

CompositeMatchResult IncrementalOnlyPrefixPolicy::match(
    const PrefixCacheMatchContext& context) {
  CompositeMatchResult result;
  CHECK(context.kv_state != nullptr);

  CacheGroupState* state = context.kv_state->group_state(c1_.state_id);
  CHECK(state != nullptr) << "C1 group state must exist before match()";

  const MMData empty_mm;
  const MMData& mm_data = context.mm_data ? *context.mm_data : empty_mm;

  // Shared blocks the sequence already holds for this group (chunked-prefill
  // re-match); usually empty on the first prefill. match() prepends them.
  const Slice<Block> existed =
      Slice<Block>(state->blocks).slice(0, state->shared_blocks_num);
  std::vector<Block> matched =
      c1_.cache->match(context.tokens, existed, mm_data);

  const size_t matched_tokens = matched.size() * c1_.block_size;
  state->shared_blocks_num = matched.size();
  state->prefix_cached_tokens = matched_tokens;
  state->blocks = matched;

  result.matched_tokens = matched_tokens;
  CompositeGroupMatch group_match;
  group_match.state_id = c1_.state_id;
  group_match.group = c1_.group;
  group_match.blocks = std::move(matched);
  result.group_matches.push_back(std::move(group_match));
  return result;
}

PrefixCacheInsertResult IncrementalOnlyPrefixPolicy::flush(
    const PrefixCacheFlushContext& context) {
  PrefixCacheInsertResult result;
  CHECK(context.kv_state != nullptr);
  CHECK(context.hash_state != nullptr);

  CacheGroupState* state = context.kv_state->group_state(c1_.state_id);
  if (state == nullptr) {
    return result;
  }

  const uint32_t stride = c1_.block_size;
  // Last full-block boundary within committed tokens, bounded by the blocks the
  // sequence actually holds and by the tokens slice we were handed.
  size_t token_end = (context.committed_tokens / stride) * stride;
  token_end =
      std::min(token_end, state->blocks.size() * static_cast<size_t>(stride));
  token_end = std::min(token_end, (context.tokens.size() / stride) * stride);

  const size_t already = state->prefix_cached_tokens;
  if (token_end <= already) {
    return result;
  }

  const MMData empty_mm;
  const MMData& mm_data = context.mm_data ? *context.mm_data : empty_mm;
  context.hash_state->ensure(
      c1_.group, stride, context.tokens, token_end, mm_data);

  const size_t first_block = already / stride;
  const size_t last_block = token_end / stride;  // exclusive
  CHECK_LE(last_block, state->blocks.size());

  // Stamp each newly-completed block with its cumulative prefix hash; the cache
  // then inserts by hash and reports the keys it had not seen before.
  std::vector<Block> flush_blocks;
  flush_blocks.reserve(last_block - first_block);
  std::vector<size_t> token_ends;
  token_ends.reserve(last_block - first_block);
  for (size_t b = first_block; b < last_block; ++b) {
    const size_t block_end = (b + 1) * static_cast<size_t>(stride);
    const XXH3Key& key = context.hash_state->key(c1_.group, block_end);
    state->blocks[b].set_hash_value(key.data);
    flush_blocks.push_back(state->blocks[b]);
    token_ends.push_back(block_end);
  }

  Slice<Block> flush_slice(flush_blocks);
  std::vector<XXH3Key> inserted_keys;
  c1_.cache->insert(flush_slice, &inserted_keys);

  result.inserted_blocks.reserve(inserted_keys.size());
  for (const XXH3Key& key : inserted_keys) {
    for (size_t i = 0; i < flush_blocks.size(); ++i) {
      if (key == XXH3Key(flush_blocks[i].get_immutable_hash_value())) {
        PrefixCacheInsertedBlock inserted;
        inserted.group = c1_.group;
        inserted.role = context.role;
        inserted.device_dp_rank = context.device_dp_rank;
        inserted.block = flush_blocks[i];
        inserted.hash_key = key;
        inserted.token_end = token_ends[i];
        inserted.hash_stride = stride;
        inserted.insert_kind = PrefixCacheInsertKind::FULL_BLOCK;
        result.inserted_blocks.push_back(std::move(inserted));
        break;
      }
    }
  }

  state->prefix_cached_tokens = token_end;
  return result;
}

}  // namespace xllm
