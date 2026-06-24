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

#include "linear_state_block_manager.h"

#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <utility>

#include "framework/model/model_input_params.h"
#include "framework/prefix_cache/block_hasher.h"
#include "framework/request/sequence.h"

namespace xllm {

namespace {

BlockManager::Options make_linear_state_options(uint32_t num_slots) {
  BlockManager::Options options;
  options.num_blocks(num_slots);
  options.block_size(/*one slot id == one block=*/1);
  options.enable_prefix_cache(false);
  options.enable_disagg_pd(false);
  options.block_type(BlockType::LINEAR);
  return options;
}

// All saves go through promotion: the sequence keeps a fresh live slot and the
// just-finished slot is grafted into the prefix cache directly. This works at
// every prefill checkpoint boundary because the slot's contents are frozen
// from the previous step and not yet overwritten by the current one.
bool can_promote_linear_state_checkpoint(const Sequence* sequence,
                                         const LinearStateCacheOp& cache_op) {
  if (sequence == nullptr || !sequence->has_linear_state_slot()) {
    return false;
  }
  if (cache_op.linear_state_id < 0 ||
      cache_op.linear_state_id != sequence->get_linear_state_slot_id()) {
    return false;
  }
  return true;
}

}  // namespace

LinearStateCheckpointReservations::Promotion::Promotion(const XXH3Key& hash,
                                                        Sequence* sequence,
                                                        int32_t live_slot_id,
                                                        Block replacement_slot)
    : hash_(hash),
      sequence_(sequence),
      live_slot_id_(live_slot_id),
      replacement_slot_(std::move(replacement_slot)) {}

LinearStateBlockManager::LinearStateBlockManager(uint32_t num_slots,
                                                 int32_t kv_block_size)
    : BlockManagerImpl(make_linear_state_options(num_slots)),
      kv_block_size_(kv_block_size) {
  CHECK_GT(num_slots, 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(kv_block_size_, 0)
      << "linear-state leaf needs the KV block size for safe-prefix clamping";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  // One live slot per sequence (mirrors SingleBlockManager). Reused until the
  // sequence finishes; subsequent calls are no-ops.
  if (seq->has_linear_state_slot()) {
    return std::vector<Block>{};
  }
  Block slot = allocate_live_slot();
  if (!slot.is_valid()) {
    LOG(ERROR) << "Failed to acquire linear state slot! free="
               << num_free_blocks() << ", used=" << num_used_blocks()
               << ", total=" << num_total_blocks();
    return std::nullopt;
  }
  std::vector<Block> blocks;
  blocks.emplace_back(std::move(slot));
  return blocks;
}

Block LinearStateBlockManager::allocate() {
  // Used by save promotion to get the sequence's next live slot. Reclaims an
  // LRU checkpoint when the free list is empty; returns an invalid Block when
  // every slot is live or pinned by an in-flight restore.
  return allocate_live_slot();
}

std::vector<Block> LinearStateBlockManager::allocate_shared(
    const Slice<int32_t>& /*token_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& /*block_hashes*/) {
  // The linear leaf has its own hash domain and never serves KV-style shared
  // blocks. The composite routes its prefix-cache shared path to the KV leaf
  // and clamps the result via compute_safe_shared_prefix_length().
  NOT_IMPLEMENTED();
  return {};
}

void LinearStateBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                                    std::vector<Block>& /*blocks*/,
                                    size_t /*existed_shared_blocks_num*/,
                                    const MMData& /*mm_data*/,
                                    const Slice<XXH3Key>& /*block_hashes*/) {
  // Linear-state checkpoints are inserted via insert_checkpoint() under the
  // leaf's own hash domain. The KV-style cache path is not meaningful here.
  NOT_IMPLEMENTED();
}

void LinearStateBlockManager::cache(const std::vector<Block>& /*blocks*/) {
  NOT_IMPLEMENTED();
}

void LinearStateBlockManager::reset_prefix_cache() {
  // Drop every committed checkpoint; the underlying free list reclaims the
  // slots as the CacheEntry Block handles destruct.
  cached_slots_.clear();
  lru_.clear();
}

Block LinearStateBlockManager::match(const XXH3Key& prefix_hash) {
  auto it = cached_slots_.find(prefix_hash);
  if (it == cached_slots_.end()) {
    return Block();
  }
  touch(it->second);
  return it->second.block;
}

int32_t LinearStateBlockManager::insert_checkpoint(const XXH3Key& prefix_hash,
                                                   Block checkpoint_block) {
  if (!checkpoint_block.is_valid()) {
    return -1;
  }

  auto it = cached_slots_.find(prefix_hash);
  if (it != cached_slots_.end()) {
    touch(it->second);
    return it->second.block.id();
  }

  const int32_t slot_id = checkpoint_block.id();
  lru_.push_back(prefix_hash);
  cached_slots_.emplace(
      prefix_hash,
      CacheEntry{std::move(checkpoint_block), std::prev(lru_.end())});
  return slot_id;
}

bool LinearStateBlockManager::contains(const XXH3Key& prefix_hash) const {
  return cached_slots_.find(prefix_hash) != cached_slots_.end();
}

LinearStateCheckpointReservations LinearStateBlockManager::resolve_cache_ops(
    std::vector<LinearStateCacheOp>* cache_ops,
    const std::vector<Sequence*>& sequences) {
  LinearStateCheckpointReservations checkpoint_reservations;
  if (cache_ops == nullptr || cache_ops->empty()) {
    return checkpoint_reservations;
  }

  // Restores are matched before saves reserve slots, so save-side eviction
  // cannot reclaim checkpoints needed by this batch's restores.
  const bool has_aligned_sequences = sequences.size() == cache_ops->size();
  for (LinearStateCacheOp& cache_op : *cache_ops) {
    if (is_zero_prefix_hash(cache_op.restore_prefix_hash)) {
      continue;
    }
    Block restore_pin = match(XXH3Key(cache_op.restore_prefix_hash.data()));
    if (restore_pin.is_valid()) {
      cache_op.restore_src_slot_id = restore_pin.id();
      checkpoint_reservations.restore_pins.emplace_back(std::move(restore_pin));
    }
  }

  // Saves are always handled by promotion: the sequence keeps a fresh live
  // slot and the old slot is grafted into the prefix cache. The actual slot
  // swap is deferred until commit_reservations() (after this step's forward),
  // because the model in this step still writes the old slot and we need its
  // end-of-step contents to become the frozen checkpoint.
  std::unordered_map<XXH3Key, int32_t, FixedStringKeyHash, FixedStringKeyEqual>
      promoted_hashes;
  for (size_t i = 0; i < cache_ops->size(); ++i) {
    LinearStateCacheOp& cache_op = (*cache_ops)[i];
    if (is_zero_prefix_hash(cache_op.save_prefix_hash)) {
      continue;
    }
    const XXH3Key save_hash(cache_op.save_prefix_hash.data());
    auto promoted_it = promoted_hashes.find(save_hash);
    if (promoted_it != promoted_hashes.end()) {
      cache_op.save_dst_slot_id = promoted_it->second;
      continue;
    }
    if (contains(save_hash)) {
      continue;
    }

    Sequence* sequence = has_aligned_sequences ? sequences[i] : nullptr;
    if (!can_promote_linear_state_checkpoint(sequence, cache_op)) {
      continue;
    }
    Block replacement_slot = allocate_live_slot();
    if (!replacement_slot.is_valid()) {
      // No free slot for the sequence's next live slot; skip this save. The
      // cache stays sparse rather than evicting under pressure.
      continue;
    }
    const int32_t live_slot_id = cache_op.linear_state_id;
    cache_op.save_dst_slot_id = live_slot_id;
    promoted_hashes.emplace(save_hash, live_slot_id);
    checkpoint_reservations.promotions.emplace_back(
        save_hash, sequence, live_slot_id, std::move(replacement_slot));
  }
  return checkpoint_reservations;
}

void LinearStateBlockManager::commit_reservations(
    LinearStateCheckpointReservations&& checkpoint_reservations) {
  for (LinearStateCheckpointReservations::Promotion& promotion :
       checkpoint_reservations.promotions) {
    Sequence* sequence = promotion.sequence();
    DCHECK(sequence != nullptr && sequence->has_linear_state_slot() &&
           sequence->get_linear_state_slot_id() == promotion.live_slot_id());
    if (sequence == nullptr || !sequence->has_linear_state_slot() ||
        sequence->get_linear_state_slot_id() != promotion.live_slot_id()) {
      continue;
    }
    Block old_live_slot = sequence->reset_linear_state_slot();
    insert_checkpoint(promotion.hash(), std::move(old_live_slot));
    sequence->set_linear_state_slot(promotion.take_replacement_slot());
    sequence->reset_linear_state_initialized();
  }
}

size_t LinearStateBlockManager::compute_safe_shared_prefix_length(
    const Sequence* sequence,
    size_t existed_shared_blocks_num,
    size_t total_shared_blocks) const {
  CHECK(sequence != nullptr);
  if (kv_block_size_ <= 0) {
    return total_shared_blocks;
  }
  const size_t block_size = static_cast<size_t>(kv_block_size_);
  size_t safe_shared_blocks =
      std::min(existed_shared_blocks_num, total_shared_blocks);
  // The last KV block of a fully tokenised prefix is the in-progress one and
  // cannot serve a linear-state restore; bound the recoverable count to all
  // *closed* blocks of the sequence's current tokens.
  const size_t max_reusable_blocks =
      sequence->num_tokens() == 0 ? 0
                                  : (sequence->num_tokens() - 1) / block_size;
  const size_t reusable_limit = std::max(
      safe_shared_blocks, std::min(total_shared_blocks, max_reusable_blocks));

  const size_t token_blocks = sequence->tokens().size() / block_size;
  const size_t hash_limit = std::min(reusable_limit, token_blocks);
  const std::vector<PrefixHash> hashes = compute_linear_state_prefix_hashes(
      sequence->tokens(), block_size, hash_limit * block_size);
  for (size_t block_idx = safe_shared_blocks; block_idx < hashes.size();
       ++block_idx) {
    if (contains(XXH3Key(hashes[block_idx].data()))) {
      safe_shared_blocks = block_idx + 1;
    }
  }
  return std::min(safe_shared_blocks, total_shared_blocks);
}

void LinearStateBlockManager::touch(CacheEntry& entry) {
  // Move the node to the MRU end in place; splice keeps lru_it valid and
  // avoids a list-node realloc.
  lru_.splice(lru_.end(), lru_, entry.lru_it);
}

Block LinearStateBlockManager::allocate_live_slot() {
  if (num_free_blocks() == 0) {
    // Walk the LRU front-to-back; skip checkpoints currently pinned by a
    // restore in this batch (ref_count > 1), evict the first non-pinned one.
    // Erasing the entry drops its Block handle, which returns the slot to the
    // underlying free list synchronously.
    auto victim_it = lru_.begin();
    while (victim_it != lru_.end()) {
      auto cached_it = cached_slots_.find(*victim_it);
      CHECK(cached_it != cached_slots_.end())
          << "LRU and linear-state prefix cache out of sync";
      if (cached_it->second.block.is_shared()) {
        ++victim_it;
        continue;
      }
      lru_.erase(victim_it);
      cached_slots_.erase(cached_it);
      break;
    }
    if (num_free_blocks() == 0) {
      // Every slot is live or pinned by a pending reservation; nothing can be
      // reclaimed from the committed prefix cache.
      return Block();
    }
  }
  std::vector<Block> blocks = BlockManagerImpl::allocate(1);
  CHECK_EQ(blocks.size(), 1u) << "linear-state slot allocation failed";
  return std::move(blocks[0]);
}

}  // namespace xllm
