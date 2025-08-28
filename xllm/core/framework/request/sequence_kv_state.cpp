/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

namespace xllm {

namespace {
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

size_t KVCacheState::shared_kv_blocks_num() const {
  return num_owned_shared_blocks_;
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
}

void KVCacheState::add_kv_blocks(const std::vector<Block>& new_blocks) {
  blocks_.insert(blocks_.end(), new_blocks.begin(), new_blocks.end());
}

void KVCacheState::add_shared_kv_blocks(std::vector<Block>&& blocks,
                                        size_t current_total_num_tokens) {
  if (blocks.empty()) {
    return;
  }

  // The number of matched blocks may be fewer than the number of blocks held by
  // the sequence itself. In this case, try to replace the blocks computed by
  // the sequence with blocks from the prefix_cache and release the computed
  // blocks to save kv_cache as much as possible.
  if (blocks.size() <= blocks_.size()) {
    try_replace_unique_blocks(
        std::move(blocks), &num_owned_shared_blocks_, &blocks_);
    return;
  }

  blocks_.clear();
  num_owned_shared_blocks_ = blocks.size();
  blocks_ = std::move(blocks);

  // update the kv cache position
  size_t num_shared_tokens = blocks_.size() * blocks_[0].size();
  // It is possible that num_shared_tokens == current_total_num_tokens,
  // indicating that the exact same prompt has been received again. In this
  // case, it becomes necessary to adjust the kv cache position to the
  // previous token, allowing the model proceed. While the shared blocks
  // should be immutable ideally, but it remains safe to regenerate the kv
  // cache in this context, given the utiliztion of the exact same token.
  if (num_shared_tokens == current_total_num_tokens) {
    size_t block_size = blocks_[0].size();
    CHECK_GT(block_size, 0);
    num_shared_tokens =
        ((current_total_num_tokens - 1) / block_size) * block_size;
  }
  CHECK_LT(num_shared_tokens, current_total_num_tokens);
  // update the kv cache position
  kv_cache_tokens_num_ = num_shared_tokens;
}

size_t KVCacheState::current_max_tokens_capacity() const {
  if (blocks_.empty()) {
    return 0;
  }
  // all blocks have the same size
  const size_t block_size = blocks_[0].size();
  return blocks_.size() * block_size;
}

// returns allocated cache blocks
Slice<Block> KVCacheState::kv_blocks() const { return blocks_; }

// get the number of blocks
size_t KVCacheState::num_kv_blocks() const { return blocks_.size(); }

std::vector<int32_t> KVCacheState::kv_cache_slots(int32_t pos_start,
                                                  int32_t pos_end) {
  CHECK(!blocks_.empty()) << "no cache blocks available";

  std::vector<int32_t> slots;
  slots.reserve(pos_end - pos_start);

  const size_t block_size = blocks_[0].size();
  for (int32_t i = pos_start; i < pos_end; ++i) {
    const int32_t block_id = blocks_[i / block_size].id();
    const int32_t block_offset = i % block_size;
    slots.push_back(block_id * block_size + block_offset);
  }
  return slots;
}

void KVCacheState::set_transfer_kv_info(TransferKVInfo&& info) {
  transfer_kv_info_ = std::move(info);
}

std::optional<TransferKVInfo>& KVCacheState::transfer_kv_info() {
  return transfer_kv_info_;
}

void KVCacheState::reset() {
  kv_cache_tokens_num_ = 0;
  num_owned_shared_blocks_ = 0;
  blocks_.clear();
  transfer_kv_info_.reset();
}

}  // namespace xllm
