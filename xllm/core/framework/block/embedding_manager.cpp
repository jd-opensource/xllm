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

#include "embedding_manager.h"

#include <glog/logging.h>

namespace xllm {
namespace {

BlockManager::Options make_embedding_options(uint32_t num_ids) {
  BlockManager::Options options;
  options.num_blocks(num_ids);
  options.block_size(/*unused=*/1);
  options.enable_prefix_cache(false);
  options.enable_disagg_pd(false);
  options.enable_cache_upload(false);
  return options;
}

}  // namespace

EmbeddingManager::EmbeddingManager(uint32_t num_ids)
    : BlockManager(make_embedding_options(num_ids)) {
  in_use_ids_.resize(num_ids, false);
  for (uint32_t id = 0; id < num_ids; ++id) {
    free_ids_.push_back(static_cast<int32_t>(num_ids - id - 1));
  }
  num_free_blocks_.store(num_ids, std::memory_order_relaxed);
}

std::vector<Block> EmbeddingManager::allocate(size_t num_blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (num_blocks > num_free_blocks_.load(std::memory_order_relaxed)) {
    return {};
  }

  std::vector<Block> blocks;
  blocks.reserve(num_blocks);
  for (size_t i = 0; i < num_blocks; ++i) {
    size_t prev_count =
        num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
    const int32_t block_id = free_ids_[prev_count - 1];
    CHECK_GE(block_id, 0);
    CHECK_LT(static_cast<size_t>(block_id), in_use_ids_.size());
    CHECK(!in_use_ids_[block_id])
        << "embedding id " << block_id << " was allocated repeatedly";
    in_use_ids_[block_id] = true;
    blocks.emplace_back(block_id, this);
  }
  num_used_blocks_.fetch_add(num_blocks, std::memory_order_relaxed);
  return blocks;
}

Block EmbeddingManager::allocate() {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_GT(num_free_blocks_.load(std::memory_order_relaxed), 0)
      << "No more embedding ids available";
  size_t prev_count = num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
  const int32_t block_id = free_ids_[prev_count - 1];
  CHECK_GE(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), in_use_ids_.size());
  CHECK(!in_use_ids_[block_id])
      << "embedding id " << block_id << " was allocated repeatedly";
  in_use_ids_[block_id] = true;
  num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
  return {block_id, this};
}

void EmbeddingManager::deallocate(const Slice<Block>& blocks) {
  for (const auto& block : blocks) {
    // EmbeddingManager does not have prefix-cache references; decrement
    // usage only when this is the last live reference owned by sequences.
    if (block.is_valid() && block.ref_count() == 1) {
      CHECK_GT(num_used_blocks_.load(std::memory_order_relaxed), 0);
      num_used_blocks_.fetch_sub(1, std::memory_order_relaxed);
    }
  }
}

std::vector<Block> EmbeddingManager::allocate_shared(
    const Slice<int32_t>& /*tokens_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/) {
  return {};
}

void EmbeddingManager::cache(const Slice<int32_t>& /*token_ids*/,
                             std::vector<Block>& /*blocks*/,
                             size_t /*existed_shared_blocks_num*/) {}

void EmbeddingManager::cache(const std::vector<Block>& /*blocks*/) {}

void EmbeddingManager::get_merged_kvcache_event(KvCacheEvent* /*event*/) const {
}

double EmbeddingManager::kv_cache_utilization() const {
  const size_t total = num_total_blocks();
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(num_used_blocks_.load(std::memory_order_relaxed)) /
         static_cast<double>(total);
}

void EmbeddingManager::free(int32_t block_id) {
  if (block_id < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_LT(static_cast<size_t>(block_id), in_use_ids_.size());
  CHECK(in_use_ids_[block_id])
      << "embedding id " << block_id << " was deallocated repeatedly";
  in_use_ids_[block_id] = false;

  size_t prev_count = num_free_blocks_.fetch_add(1, std::memory_order_relaxed);
  CHECK_LT(prev_count, free_ids_.size());
  free_ids_[prev_count] = block_id;
}

}  // namespace xllm
