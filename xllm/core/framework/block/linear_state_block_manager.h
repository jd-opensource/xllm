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
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "block_manager_impl.h"
#include "common/macros.h"
#include "util/hash_util.h"

namespace xllm {

class Sequence;
struct LinearStateCacheOp;

// Aggregated batch state produced by LinearStateBlockManager::resolve_cache_ops
// and consumed by ::commit_reservations after this step's forward finishes.
// Saves are handled entirely by promotion (no extra device copy), mirroring
// how KV-cache blocks are inserted into the prefix cache without a copy.
struct LinearStateCheckpointReservations {
  // A live slot the scheduler decided to "promote" into a committed
  // checkpoint: the sequence keeps a fresh replacement live slot and the
  // existing live slot is grafted directly into the prefix cache after this
  // step's forward writes its end-of-step contents.
  class Promotion final {
   public:
    Promotion(const XXH3Key& hash,
              Sequence* sequence,
              int32_t live_slot_id,
              Block replacement_slot);
    MOVE_ONLY(Promotion);

    const XXH3Key& hash() const { return hash_; }
    Sequence* sequence() const { return sequence_; }
    int32_t live_slot_id() const { return live_slot_id_; }
    Block take_replacement_slot() { return std::move(replacement_slot_); }

   private:
    XXH3Key hash_;
    Sequence* sequence_ = nullptr;
    int32_t live_slot_id_ = -1;
    Block replacement_slot_;
  };

  int32_t dp_rank = -1;
  // Pin matched checkpoints until the worker copies them into sequence-owned
  // live slots on the compute stream.
  std::vector<Block> restore_pins;
  std::vector<Promotion> promotions;
};

// CompositeBlockManager leaf for Qwen3.5 GDN linear-state checkpoints.
// Registered under BlockType::LINEAR.
//
// Inherits BlockManagerImpl for the slot id pool (free_blocks_, padding id 0,
// allocate/free, num_* accounting). Disables the base prefix_cache_
// (PrefixCache is KV-shaped and lives in the KV hash domain) and overlays its
// own bespoke LRU + XXH3Key -> Block index. Hash domain is INDEPENDENT of the
// KV leaf -- keys come from compute_linear_state_prefix_hashes(seq->tokens(),
// block_size, boundary), NOT from Block::get_immutable_hash_value() or
// seq->block_hashes().
//
// The id pool is unified: slot ids [1, num_slots) serve LIVE slots (held by a
// sequence under composite_blocks_[LINEAR]) AND committed CHECKPOINT slots
// (pinned by cached_slots_ in this leaf) interchangeably under reference
// counting. Slot 0 is the inherited padding block.
class LinearStateBlockManager final : public BlockManagerImpl {
 public:
  // num_slots:     total physical slots [0, N); id 0 padding (inherited),
  //                [1, N) usable. CHECK: N >= 2.
  // kv_block_size: the KV-side block_size (tokens per block); used by the
  //                clamp helper to compute linear hash boundaries. The
  //                leaf's own BlockManagerImpl::block_size is 1.
  LinearStateBlockManager(uint32_t num_slots, int32_t kv_block_size);
  ~LinearStateBlockManager() override = default;

  // ---- BlockManagerImpl overrides ----

  // One live slot per sequence (mirrors SingleBlockManager). Returns empty
  // when the sequence already holds a LINEAR slot; nullopt when no slot is
  // free AND no LRU checkpoint can be reclaimed. Composite commits under
  // BlockType::LINEAR.
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      size_t num_tokens) override;

  // Bring base allocate(size_t) into scope so the base ctor's padding
  // reservation keeps working (name-hiding fix; same pattern as
  // SingleBlockManager).
  using BlockManagerImpl::allocate;
  // Single-slot allocate with LRU-eviction fallback. Used by save promotion.
  Block allocate() override;

  // KV-style prefix surface does NOT apply to this leaf (different hash
  // domain). Mirrors SingleBlockManager's NOT_IMPLEMENTED pattern.
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
  void cache(const std::vector<Block>& blocks) override;

  size_t num_blocks_in_prefix_cache() const override {
    return cached_slots_.size();
  }
  void reset_prefix_cache() override;

  // ---- Linear-state public API (own hash domain, called through the
  // composite's typed accessor; NOT routed via supports_prefix_cache). ----

  Block match(const XXH3Key& prefix_hash);
  int32_t insert_checkpoint(const XXH3Key& prefix_hash, Block checkpoint_block);
  bool contains(const XXH3Key& prefix_hash) const;

  // Resolve this batch's restore + save plan in this leaf's hash domain.
  // dp_rank is stamped by the pool (composite/leaf do not know their rank).
  LinearStateCheckpointReservations resolve_cache_ops(
      std::vector<LinearStateCacheOp>* cache_ops,
      const std::vector<Sequence*>& sequences = {});

  // Apply post-forward promotions: graft old live slots into cached_slots_;
  // hand replacement slots to sequences (mark as uninitialized).
  void commit_reservations(LinearStateCheckpointReservations&& reservations);

  // Count of shared KV blocks for which a linear-state checkpoint exists.
  // Caller deallocates KV blocks beyond this count.
  size_t compute_safe_shared_prefix_length(const Sequence* sequence,
                                           size_t existed_shared_blocks_num,
                                           size_t total_shared_blocks) const;

  int32_t kv_block_size() const { return kv_block_size_; }

  // LRU-evicting single-slot allocate: walks lru_ front->back skipping
  // ref_count > 1 (pinned by an in-flight restore this batch), evicts the
  // first non-pinned entry by erasing the map+LRU entry (Block dtor returns
  // the slot to the inherited free list), then BlockManagerImpl::allocate(1).
  // Public so unit tests and the save-promotion path can request a fresh live
  // slot under the same eviction policy used internally by allocate().
  Block allocate_live_slot();

 private:
  struct CacheEntry {
    Block block;
    std::list<XXH3Key>::iterator lru_it;
  };

  // Move an existing checkpoint to the most-recently-used end of the LRU.
  void touch(CacheEntry& entry);

  int32_t kv_block_size_;
  std::unordered_map<XXH3Key,
                     CacheEntry,
                     FixedStringKeyHash,
                     FixedStringKeyEqual>
      cached_slots_;
  std::list<XXH3Key> lru_;  // front = least recently used
};

}  // namespace xllm
