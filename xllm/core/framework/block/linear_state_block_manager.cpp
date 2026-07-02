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
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/config/scheduler_config.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"

namespace xllm {

namespace {

BlockManager::Options make_linear_state_options(uint32_t num_slots,
                                                int32_t kv_block_size) {
  BlockManager::Options options;
  options.num_blocks(num_slots);
  // The slot pool is one id per slot; block_size carries the real KV block
  // size so the checkpoint index can convert its KV-block budget into whole
  // prefill chunks without the leaf holding a private copy.
  options.block_size(kv_block_size);
  options.enable_prefix_cache(true);
  options.enable_disagg_pd(false);
  options.block_type(BlockType::LINEAR);
  return options;
}

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

LinearStateBlockManager::LinearStateBlockManager(uint32_t num_slots,
                                                 int32_t kv_block_size)
    : BlockManagerImpl(make_linear_state_options(num_slots, kv_block_size)) {
  CHECK_GT(num_slots, 1u)
      << "linear-state leaf needs at least one usable slot (plus padding)";
  CHECK_GT(kv_block_size, 0)
      << "linear-state leaf needs the KV block size for safe-prefix clamping";
}

std::optional<std::vector<Block>>
LinearStateBlockManager::allocate_for_sequence(Sequence* seq,
                                               size_t /*num_tokens*/) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  if (seq->has_linear_state_slot()) {
    return std::vector<Block>{};
  }
  Block slot = allocate();
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
  std::vector<Block> blocks = BlockManagerImpl::allocate(1);
  if (blocks.empty()) {
    return Block();
  }
  return std::move(blocks[0]);
}

std::vector<Block> LinearStateBlockManager::allocate_shared(
    const Slice<int32_t>& /*token_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/,
    const MMData& /*mm_data*/,
    const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
  return {};
}

void LinearStateBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                                    std::vector<Block>& /*blocks*/,
                                    size_t /*existed_shared_blocks_num*/,
                                    const MMData& /*mm_data*/,
                                    const Slice<XXH3Key>& /*block_hashes*/) {
  NOT_IMPLEMENTED();
}

Block LinearStateBlockManager::match(const XXH3Key& prefix_hash) {
  return prefix_cache_->find(prefix_hash);
}

bool LinearStateBlockManager::contains(const XXH3Key& prefix_hash) const {
  return prefix_cache_->contains(prefix_hash);
}

void LinearStateBlockManager::apply_pending_saves(
    const std::vector<Sequence*>& sequences) {
  for (Sequence* seq : sequences) {
    if (seq == nullptr || !seq->has_pending_linear_save()) {
      continue;
    }
    std::optional<XXH3Key> maybe_hash = seq->take_pending_linear_save();
    if (!maybe_hash) {
      continue;
    }
    const XXH3Key& hash = *maybe_hash;
    if (prefix_cache_->contains(hash) || !seq->has_linear_state_slot()) {
      continue;
    }
    Block new_live_slot = allocate();
    if (!new_live_slot.is_valid()) {
      continue;
    }
    // Promote the sequence's current (warm) slot into the checkpoint index via
    // the inherited pre-hashed cache() primitive, then hand the sequence a
    // fresh live slot. copy_block() takes a handle copy (not a move) so the
    // slot lives in both the sequence and the checkpoint index until
    // replace_block() swaps it out -- the same copy-on-write shape KV blocks
    // use.
    Block old_copy = seq->copy_block(BlockType::LINEAR);
    old_copy.set_hash_value(hash.data);
    std::vector<Block> checkpoint;
    checkpoint.emplace_back(std::move(old_copy));
    cache(checkpoint);
    seq->replace_block(BlockType::LINEAR, std::move(new_live_slot));
    seq->reset_linear_state_initialized();
  }
}

void LinearStateBlockManager::resolve_cache_ops(
    std::vector<LinearStateCacheOp>* cache_ops,
    const std::vector<Sequence*>& sequences) {
  if (cache_ops == nullptr || cache_ops->empty()) {
    return;
  }

  for (LinearStateCacheOp& cache_op : *cache_ops) {
    if (is_zero_prefix_hash(cache_op.restore_prefix_hash)) {
      continue;
    }
    Block matched =
        prefix_cache_->find(XXH3Key(cache_op.restore_prefix_hash.data()));
    if (matched.is_valid()) {
      cache_op.restore_src_slot_id = matched.id();
    }
  }

  const bool has_aligned_sequences = sequences.size() == cache_ops->size();
  std::unordered_map<XXH3Key, int32_t, FixedStringKeyHash, FixedStringKeyEqual>
      saved_hashes;
  for (size_t i = 0; i < cache_ops->size(); ++i) {
    LinearStateCacheOp& cache_op = (*cache_ops)[i];
    if (is_zero_prefix_hash(cache_op.save_prefix_hash)) {
      continue;
    }
    const XXH3Key save_hash(cache_op.save_prefix_hash.data());
    auto saved_it = saved_hashes.find(save_hash);
    if (saved_it != saved_hashes.end()) {
      cache_op.save_dst_slot_id = saved_it->second;
      continue;
    }
    if (prefix_cache_->contains(save_hash)) {
      continue;
    }

    Sequence* sequence = has_aligned_sequences ? sequences[i] : nullptr;
    if (!can_promote_linear_state_checkpoint(sequence, cache_op)) {
      continue;
    }
    const int32_t live_slot_id = cache_op.linear_state_id;
    cache_op.save_dst_slot_id = live_slot_id;
    saved_hashes.emplace(save_hash, live_slot_id);
    sequence->set_pending_linear_save(save_hash);
  }
}

size_t LinearStateBlockManager::recoverable_shared_prefix_blocks(
    const Sequence* sequence,
    size_t existed_shared_blocks_num,
    size_t max_kv_shared_blocks) const {
  CHECK(sequence != nullptr);
  const int32_t kv_block_size = static_cast<int32_t>(block_size());
  const int32_t chunk_stride_tokens = ::xllm::SchedulerConfig::get_instance()
                                          .max_tokens_per_chunk_for_prefill();
  if (kv_block_size <= 0) {
    return max_kv_shared_blocks;
  }
  if (chunk_stride_tokens <= 0 || chunk_stride_tokens % kv_block_size != 0) {
    return max_kv_shared_blocks;
  }
  const size_t block_size = static_cast<size_t>(kv_block_size);
  const size_t chunk_stride = static_cast<size_t>(chunk_stride_tokens);
  const size_t blocks_per_chunk = chunk_stride / block_size;
  const size_t safe_shared_blocks =
      std::min(existed_shared_blocks_num, max_kv_shared_blocks);
  const size_t max_reusable_blocks =
      sequence->num_tokens() == 0 ? 0
                                  : (sequence->num_tokens() - 1) / block_size;
  const size_t reusable_limit = std::max(
      safe_shared_blocks, std::min(max_kv_shared_blocks, max_reusable_blocks));

  // Convert the KV-block budget into whole prefill chunks, then let the
  // checkpoint index own the hashing + probe over its own (chunk-strided) hash
  // domain. The leaf only supplies the sequence bounds; the index counts in KV
  // blocks and the composite takes the final min(kv_match, .).
  const size_t token_chunks = sequence->tokens().size() / chunk_stride;
  const size_t reusable_chunks = reusable_limit / blocks_per_chunk;
  const size_t probe_chunks = std::min(reusable_chunks, token_chunks);
  auto* checkpoint_index =
      static_cast<LinearStatePrefixCache*>(prefix_cache_.get());
  // max_kv_shared_blocks bounds the probe range above, so the result is already
  // <= it; the composite takes the final min(kv_match, .) across leaves.
  return checkpoint_index->recoverable_prefix_blocks(sequence->tokens(),
                                                     chunk_stride,
                                                     blocks_per_chunk,
                                                     safe_shared_blocks,
                                                     probe_chunks);
}

}  // namespace xllm
