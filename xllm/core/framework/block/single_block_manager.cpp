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

#include "single_block_manager.h"

#include <glog/logging.h>

#include <utility>

namespace xllm {
namespace {

BlockManager::Options make_single_block_options(uint32_t num_blocks) {
  BlockManager::Options options;
  options.num_blocks(num_blocks);
  options.block_size(/*unused=*/1);
  options.enable_prefix_cache(false);
  options.enable_disagg_pd(false);
  options.enable_cache_upload(false);
  return options;
}

}  // namespace

SingleBlockManager::SingleBlockManager(uint32_t num_blocks,
                                       std::string resource_name,
                                       std::string exhaustion_message)
    : BlockManager(make_single_block_options(num_blocks)),
      resource_name_(std::move(resource_name)),
      exhaustion_message_(std::move(exhaustion_message)) {
  CHECK_GT(num_blocks, 0) << "No blocks to allocate";
  in_use_ids_.resize(num_blocks, false);
  usage_accounted_ids_.resize(num_blocks, false);
  free_ids_.reserve(num_blocks);
  for (uint32_t id = 0; id < num_blocks; ++id) {
    free_ids_.push_back(static_cast<int32_t>(num_blocks - id - 1));
  }
  num_free_blocks_.store(num_blocks, std::memory_order_relaxed);
}

std::vector<Block> SingleBlockManager::allocate(size_t num_blocks) {
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
        << resource_name_ << " id " << block_id << " was allocated repeatedly";
    in_use_ids_[block_id] = true;
    usage_accounted_ids_[block_id] = true;
    blocks.emplace_back(block_id, this);
  }
  num_used_blocks_.fetch_add(num_blocks, std::memory_order_relaxed);
  return blocks;
}

Block SingleBlockManager::allocate() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (exhaustion_message_.empty()) {
    CHECK_GT(num_free_blocks_.load(std::memory_order_relaxed), 0)
        << "No more " << resource_name_ << " blocks available";
  } else {
    CHECK_GT(num_free_blocks_.load(std::memory_order_relaxed), 0)
        << exhaustion_message_;
  }
  size_t prev_count = num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
  const int32_t block_id = free_ids_[prev_count - 1];
  CHECK_GE(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), in_use_ids_.size());
  CHECK(!in_use_ids_[block_id])
      << resource_name_ << " id " << block_id << " was allocated repeatedly";
  in_use_ids_[block_id] = true;
  usage_accounted_ids_[block_id] = true;
  num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
  return {block_id, this};
}

void SingleBlockManager::deallocate(const Slice<Block>& blocks) {
  for (const auto& block : blocks) {
    if (!block.is_valid()) {
      continue;
    }
    const int32_t block_id = block.id();
    if (block_id < 0) {
      continue;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK_LT(static_cast<size_t>(block_id), usage_accounted_ids_.size());

    // Drop effective usage only when the deallocated reference is the last live
    // `Block` reference. If shared aliases exist, `free()` will reconcile usage
    // when the last alias releases the id.
    if (usage_accounted_ids_[block_id] && block.ref_count() == 1) {
      usage_accounted_ids_[block_id] = false;
      CHECK_GT(num_used_blocks_.load(std::memory_order_relaxed), 0u);
      num_used_blocks_.fetch_sub(1, std::memory_order_relaxed);
    }
  }
}

std::vector<Block> SingleBlockManager::allocate_shared(
    const Slice<int32_t>& /*tokens_ids*/,
    const Slice<Block>& /*existed_shared_blocks*/) {
  // Prefix cache is disabled in this manager. Keep it substitutable with
  // BlockManager behavior under enable_prefix_cache=false.
  return {};
}

void SingleBlockManager::cache(const Slice<int32_t>& /*token_ids*/,
                               std::vector<Block>& /*blocks*/,
                               size_t /*existed_shared_blocks_num*/) {
  // Prefix cache is disabled in this manager: no-op.
}

void SingleBlockManager::cache(const std::vector<Block>& /*blocks*/) {
  // Prefix cache is disabled in this manager: no-op.
}

void SingleBlockManager::get_merged_kvcache_event(
    KvCacheEvent* /*event*/) const {
  // Prefix cache is disabled in this manager: no-op.
}

size_t SingleBlockManager::num_blocks_in_prefix_cache() const { return 0; }

size_t SingleBlockManager::num_free_blocks() const { return num_free_blocks_; }

size_t SingleBlockManager::num_used_blocks() const { return num_used_blocks_; }

double SingleBlockManager::kv_cache_utilization() const {
  const size_t total = num_total_blocks();
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(num_used_blocks_.load(std::memory_order_relaxed)) /
         static_cast<double>(total);
}

void SingleBlockManager::free(int32_t block_id) {
  if (block_id < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_LT(static_cast<size_t>(block_id), in_use_ids_.size());
  CHECK(in_use_ids_[block_id])
      << resource_name_ << " id " << block_id << " was deallocated repeatedly";
  in_use_ids_[block_id] = false;

  // If `deallocate()` was skipped (or called on a different alias), converge
  // effective usage when the last `Block` reference releases the id.
  if (usage_accounted_ids_[block_id]) {
    usage_accounted_ids_[block_id] = false;
    CHECK_GT(num_used_blocks_.load(std::memory_order_relaxed), 0u);
    num_used_blocks_.fetch_sub(1, std::memory_order_relaxed);
  }

  size_t prev_count = num_free_blocks_.fetch_add(1, std::memory_order_relaxed);
  CHECK_LT(prev_count, free_ids_.size());
  free_ids_[prev_count] = block_id;
}

size_t SingleBlockManager::num_total_blocks() const { return free_ids_.size(); }

}  // namespace xllm
