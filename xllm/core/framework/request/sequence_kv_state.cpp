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

#include "sequence_kv_state.h"

#include <algorithm>

namespace xllm {

namespace {
// Empty block list returned by read views when a BlockType is absent from the
// map. Static so the returned Slice / reference stays valid.
const std::vector<Block>& empty_blocks() {
  static const std::vector<Block> kEmpty;
  return kEmpty;
}

void try_replace_unique_blocks(std::vector<Block>&& matched_shared_blocks,
                               uint32_t* num_owned_shared_blocks,
                               std::vector<Block>* owned_blocks) {
  uint32_t num_matched_shared_blocks = matched_shared_blocks.size();
  if (*num_owned_shared_blocks < num_matched_shared_blocks) {
    CHECK_GE(owned_blocks->size(), num_matched_shared_blocks);
    std::move(matched_shared_blocks.begin(),
              matched_shared_blocks.begin() + num_matched_shared_blocks,
              owned_blocks->begin());
    *num_owned_shared_blocks = num_matched_shared_blocks;
  }
}
}  // namespace

const std::vector<Block>& KVCacheState::kv_view() const {
  const auto it = composite_blocks_.find(BlockType::KV);
  return it == composite_blocks_.end() ? empty_blocks() : it->second;
}

std::vector<Block>& KVCacheState::mutable_kv_view() {
  return composite_blocks_[BlockType::KV];
}

size_t KVCacheState::shared_kv_blocks_num() const {
  return num_owned_shared_blocks_;
}

size_t KVCacheState::shared_kv_tokens_num() const {
  const std::vector<Block>& kv = kv_view();
  if (kv.empty() || num_owned_shared_blocks_ == 0) {
    return 0;
  }
  return num_owned_shared_blocks_ * kv[0].size();
}

size_t KVCacheState::kv_cache_tokens_num() const {
  return kv_cache_tokens_num_;
}

void KVCacheState::set_kv_cache_tokens_num(size_t num) {
  kv_cache_tokens_num_ = num;
}

void KVCacheState::incr_kv_cache_tokens_num(size_t num) {
  CHECK(kv_cache_tokens_num_ + num <= current_max_tokens_capacity());
  kv_cache_tokens_num_ += num;
  slice_window_pos_ += num;
}

void KVCacheState::set_slice_window_size(uint32_t size) {
  CHECK(size > 0);
  CHECK(!blocks_of(BlockType::SWA).empty());
  slice_window_size_ = size;
  slice_window_pos_ = 0;
  slice_window_buffer_ = size;
}

void KVCacheState::update_slice_window_pos() {
  if (slice_window_size_ > 0) {
    if (slice_window_pos_ >= slice_window_buffer_) {
      // Preserve the legacy cursor contract for non-composite block managers.
      // CompositeBlockManager exposes DSA SWA tables as absolute logical
      // columns instead.
      slice_window_pos_ = slice_window_pos_ % slice_window_size_;
    }
  }
}

void KVCacheState::add_kv_blocks(const std::vector<Block>& new_blocks) {
  std::vector<Block>& kv = mutable_kv_view();
  kv.insert(kv.end(), new_blocks.begin(), new_blocks.end());
}

void KVCacheState::incr_shared_kv_blocks_num(size_t num) {
  CHECK(num_owned_shared_blocks_ + num <= num_kv_blocks());
  num_owned_shared_blocks_ += num;
}

void KVCacheState::add_shared_kv_blocks(std::vector<Block>&& blocks,
                                        size_t current_total_num_tokens) {
  if (blocks.empty()) {
    return;
  }
  std::vector<Block>& kv = mutable_kv_view();
  // The number of matched blocks may be fewer than the number of blocks held by
  // the sequence itself. In this case, try to replace the blocks computed by
  // the sequence with blocks from the prefix_cache and release the computed
  // blocks to save kv_cache as much as possible.
  if (blocks.size() <= kv.size()) {
    try_replace_unique_blocks(
        std::move(blocks), &num_owned_shared_blocks_, &kv);
    return;
  }

  kv.clear();
  num_owned_shared_blocks_ = blocks.size();
  kv = std::move(blocks);

  // update the kv cache position
  size_t num_shared_tokens = kv.size() * kv[0].size();
  // It is possible that num_shared_tokens == current_total_num_tokens,
  // indicating that the exact same prompt has been received again. In this
  // case, it becomes necessary to adjust the kv cache position to the
  // previous token, allowing the model proceed. While the shared blocks
  // should be immutable ideally, but it remains safe to regenerate the kv
  // cache in this context, given the utiliztion of the exact same token.
  if (num_shared_tokens == current_total_num_tokens) {
    size_t block_size = kv[0].size();
    CHECK_GT(block_size, 0);
    num_shared_tokens =
        ((current_total_num_tokens - 1) / block_size) * block_size;
    if (num_owned_shared_blocks_ > 0) {
      num_owned_shared_blocks_--;
      kv.pop_back();
    }
  }
  CHECK_LT(num_shared_tokens, current_total_num_tokens);
  // update the kv cache position
  kv_cache_tokens_num_ = num_shared_tokens;
}

size_t KVCacheState::current_max_tokens_capacity() const {
  const std::vector<Block>& kv = kv_view();
  if (!kv.empty()) {
    // all blocks have the same size
    const size_t block_size = kv[0].size();
    return kv.size() * block_size;
  }
  // DSV4: only the compressed incremental groups (C4 / C128) have a linear
  // token capacity. The SWA ring is excluded on purpose -- its committed tokens
  // keep advancing past ring_capacity * block_size, so counting it here would
  // make incr_kv_cache_tokens_num's CHECK fail.
  size_t capacity = 0;
  for (const BlockType type : {BlockType::C4, BlockType::C128}) {
    const std::vector<Block>& blocks = blocks_of(type);
    if (blocks.empty()) {
      continue;
    }
    const size_t group_capacity = blocks.size() * blocks[0].size();
    capacity =
        capacity == 0 ? group_capacity : std::min(capacity, group_capacity);
  }
  return capacity;
}

// returns allocated cache blocks
Slice<Block> KVCacheState::kv_blocks() const { return kv_view(); }

std::vector<Block>* KVCacheState::mutable_kv_blocks() {
  return &mutable_kv_view();
}

// get the number of blocks
size_t KVCacheState::num_kv_blocks() const { return kv_view().size(); }

std::vector<int32_t> KVCacheState::kv_cache_slots(int32_t pos_start,
                                                  int32_t pos_end) {
  const std::vector<Block>& kv = kv_view();
  CHECK(!kv.empty()) << "no cache blocks available";

  std::vector<int32_t> slots;
  slots.reserve(pos_end - pos_start);

  const size_t block_size = kv[0].size();
  for (int32_t i = pos_start; i < pos_end; ++i) {
    const int32_t block_id = kv[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

Slice<Block> KVCacheState::blocks_of(BlockType type) const {
  const auto it = composite_blocks_.find(type);
  return it == composite_blocks_.end() ? Slice<Block>(empty_blocks())
                                       : Slice<Block>(it->second);
}

std::vector<Block>* KVCacheState::mutable_blocks_of(BlockType type) {
  return &composite_blocks_[type];
}

std::vector<std::pair<BlockType, const std::vector<Block>*>>
KVCacheState::multi_block_export_view() const {
  std::vector<std::pair<BlockType, const std::vector<Block>*>> view;
  view.reserve(kMultiBlockExportOrder.size());
  for (const BlockType type : kMultiBlockExportOrder) {
    const auto it = composite_blocks_.find(type);
    if (it != composite_blocks_.end()) {
      view.emplace_back(type, &it->second);
    }
  }
  return view;
}

bool KVCacheState::has_multi_block_export() const {
  for (const BlockType type : kMultiBlockExportOrder) {
    if (composite_blocks_.find(type) != composite_blocks_.end()) {
      return true;
    }
  }
  return false;
}

bool KVCacheState::has_single_block() const {
  const auto it = composite_blocks_.find(BlockType::Single);
  return it != composite_blocks_.end() && !it->second.empty() &&
         it->second[0].is_valid();
}

int32_t KVCacheState::single_block_id() const {
  return has_single_block() ? composite_blocks_.at(BlockType::Single)[0].id()
                            : -1;
}

void KVCacheState::set_single_block(Block&& block) {
  std::vector<Block>& single = composite_blocks_[BlockType::Single];
  single.clear();
  single.emplace_back(std::move(block));
}

Block KVCacheState::reset_single_block() {
  const auto it = composite_blocks_.find(BlockType::Single);
  if (it == composite_blocks_.end() || it->second.empty()) {
    return Block();
  }
  Block block = std::move(it->second[0]);
  composite_blocks_.erase(it);
  return block;
}

void KVCacheState::set_transfer_kv_info(TransferKVInfo&& info) {
  transfer_kv_info_ = std::move(info);
}

std::optional<TransferKVInfo>& KVCacheState::transfer_kv_info() {
  return transfer_kv_info_;
}

size_t KVCacheState::next_transfer_block_idx() const {
  return next_transfer_block_idx_;
}

void KVCacheState::set_next_transfer_block_idx(size_t idx) {
  next_transfer_block_idx_ = idx;
}

void KVCacheState::advance_transfer_block_idx(size_t idx) {
  next_transfer_block_idx_ = std::max(next_transfer_block_idx_, idx);
}

void KVCacheState::reset() {
  kv_cache_tokens_num_ = 0;
  num_owned_shared_blocks_ = 0;
  pushed_local_block_count_ = 0;
  composite_blocks_.clear();
  transfer_kv_info_.reset();
  next_transfer_block_idx_ = 0;
}

void KVCacheState::process_beam_search(std::optional<Block> new_block) {
  std::vector<Block>& kv = mutable_kv_view();
  kv.clear();
  kv = std::move(src_blocks_);

  if (new_block.has_value()) {
    kv.pop_back();
    kv.emplace_back(new_block.value());
  }
}

}  // namespace xllm
