/* Copyright 2025-2026 The xLLM Authors.

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

#include <cstdint>
#include <optional>
#include <vector>

#include "block_manager_impl.h"
#include "util/hash_util.h"

namespace xllm {

class Sequence;
struct LinearStateCacheOp;

// CompositeBlockManager leaf for Qwen3.5 GDN linear-state checkpoints.
// Registered under BlockType::LINEAR, enable_prefix_cache=true.
//
// Inherits BlockManagerImpl for the slot id pool and prefix cache. The
// inherited prefix_cache_ serves as the checkpoint index (LRU + eviction that
// skips is_shared() slots). Hash domain is INDEPENDENT of the KV leaf — keys
// come from compute_linear_state_prefix_hashes(), not from KV block hashes.
//
// The id pool is unified: slot ids [1, num_slots) serve LIVE slots (held by a
// sequence under composite_blocks_[LINEAR]) AND committed CHECKPOINT slots
// (owned by prefix_cache_) interchangeably under reference counting. Slot 0 is
// the inherited padding block.
class LinearStateBlockManager final : public BlockManagerImpl {
 public:
  LinearStateBlockManager(uint32_t num_slots, int32_t kv_block_size);
  ~LinearStateBlockManager() override = default;

  // ---- BlockManagerImpl overrides ----

  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;

  using BlockManagerImpl::allocate;
  Block allocate() override;

  // allocate_shared() and the token-id cache() overload take the KV token-chain
  // surface, which is not meaningful for this leaf: linear-state keys come from
  // compute_linear_state_prefix_hashes(), not from token ids.
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  // The pre-hashed cache(const std::vector<Block>&) primitive is reused
  // verbatim for checkpoint insertion, so keep the base version instead of a
  // bespoke one.
  using BlockManagerImpl::cache;

  // ---- Linear-state public API ----

  Block match(const XXH3Key& prefix_hash);
  bool contains(const XXH3Key& prefix_hash) const;

  void apply_pending_saves(const std::vector<Sequence*>& sequences);
  void resolve_cache_ops(std::vector<LinearStateCacheOp>* cache_ops,
                         const std::vector<Sequence*>& sequences = {});

  // Linear-state view of how many leading KV shared blocks are recoverable from
  // a committed checkpoint. max_kv_shared_blocks bounds the probe range (and
  // saves hashing); the caller (composite) takes the final min across leaves.
  size_t recoverable_shared_prefix_blocks(const Sequence* sequence,
                                          size_t existed_shared_blocks_num,
                                          size_t max_kv_shared_blocks) const;
};

}  // namespace xllm
