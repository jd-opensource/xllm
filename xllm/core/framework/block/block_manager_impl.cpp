/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "block_manager_impl.h"

#include <unordered_set>

#include "framework/prefix_cache/prefix_cache_factory.h"
namespace xllm {

BlockManagerImpl::BlockManagerImpl(const Options& options)
    : BlockManager(options) {
  CHECK_GT(options.num_blocks(), 0) << "No blocks to allocate";
  CHECK_GT(options.block_size(), 0) << "Block size must be positive";
  if (options_.enable_prefix_cache()) {
    prefix_cache_ = create_prefix_cache(options.block_size(),
                                        options.enable_cache_upload());
    CHECK(prefix_cache_) << "Failed to create prefix cache!";
  }

  size_t total_blocks = options_.num_blocks();
  block_size_ = options_.block_size();
  num_free_blocks_.store(total_blocks, std::memory_order_relaxed);
  free_blocks_.reserve(total_blocks);
  for (int32_t i = 0; i < total_blocks; ++i) {
    // push smaller block ids to the back of the vector
    free_blocks_.push_back(total_blocks - i - 1);
  }

  // reserve block 0 for padding
  padding_block_ = allocate();
  CHECK_EQ(padding_block_.id(), 0) << "Padding block id should be 0";
}

std::vector<Block> BlockManagerImpl::allocate(size_t num_blocks) {
  if (!has_enough_blocks(num_blocks)) {
    return {};
  }

  CHECK(num_blocks <= num_free_blocks_) << "Not enough blocks available";
  std::vector<Block> blocks;
  blocks.reserve(num_blocks);
  for (uint32_t i = 0; i < num_blocks; ++i) {
    size_t prev_count =
        num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
    const int32_t block_id = free_blocks_[prev_count - 1];
    blocks.emplace_back(block_id, this);
  }

  // const auto block_ids = allocate(num_blocks);
  num_used_blocks_.fetch_add(num_blocks, std::memory_order_relaxed);
  return blocks;
}

void BlockManagerImpl::deallocate(const Slice<Block>& blocks) {
  if (options_.enable_prefix_cache()) {
    for (const auto& block : blocks) {
      // the block is not shared by other sequence
      if (block.is_valid() && block.ref_count() <= 2) {
        if (num_used_blocks_ == 0) {
          LOG(ERROR) << "num_used_blocks_==0 cannot fetch_sub for id:"
                     << block.id()
                     << ", total block size: " << num_total_blocks();
          std::unordered_set<int32_t> block_id_set;
          block_id_set.insert(block.id());
          std::string error_msg = "Block already released: ";
          for (auto& id : free_blocks_) {
            if (block_id_set.count(id) != 0) {
              error_msg.append(std::to_string(id)).append(" ");
            }
          }
          LOG(FATAL) << error_msg;
        }
        num_used_blocks_.fetch_sub(1, std::memory_order_relaxed);
      }
    }
  } else {
    num_used_blocks_.fetch_sub(blocks.size(), std::memory_order_relaxed);
  }
}

bool BlockManagerImpl::has_enough_blocks(uint32_t num_blocks) {
  if (num_blocks <= num_free_blocks_) {
    return true;
  }

  // prefix cache is disabled, no way to evict blocks
  if (!options_.enable_prefix_cache()) {
    return false;
  }

  // try to evict some blocks from the prefix cache
  const uint32_t n_blocks_to_evict = num_blocks - num_free_blocks_;

  AUTO_COUNTER(prefix_cache_latency_seconds_evict);
  const uint32_t n_blocks_evicted = prefix_cache_->evict(n_blocks_to_evict);
  if (n_blocks_evicted < n_blocks_to_evict) {
    return false;
  }

  if (num_free_blocks_ >= num_blocks) {
    return true;
  }

  LOG(WARNING) << "Potential block leak, free blocks in allocator: "
               << num_free_blocks_
               << " blocks in prefix cache: " << prefix_cache_->num_blocks();
  return false;
}

std::vector<Block> BlockManagerImpl::allocate_shared(
    const Slice<int32_t>& tokens_ids,
    const Slice<Block>& existed_shared_blocks) {
  // only allocate shared blocks for prefill sequences
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_match);

    std::vector<Block> shared_blocks =
        prefix_cache_->match(tokens_ids, existed_shared_blocks);

    const size_t prefix_length =
        shared_blocks.empty() ? 0
                              : shared_blocks.size() * shared_blocks[0].size();
    COUNTER_ADD(prefix_cache_match_length_total, prefix_length);

    // update effective block usage
    for (const auto& block : shared_blocks) {
      // the block is not shared by any sequence
      if (block.ref_count() <= 2) {
        num_used_blocks_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    return shared_blocks;
  }
  return {};
}

void BlockManagerImpl::cache(const Slice<int32_t>& token_ids,
                             std::vector<Block>& blocks) {
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_insert);
    // Add the kv cache to the prefix cache
    prefix_cache_->insert(token_ids, blocks);
  }
}

void BlockManagerImpl::cache(const std::vector<Block>& blocks) {
  if (options_.enable_prefix_cache()) {
    AUTO_COUNTER(prefix_cache_latency_seconds_insert);
    // Add the kv cache to the prefix cache
    prefix_cache_->insert(blocks);
  }
}

void BlockManagerImpl::get_merged_kvcache_event(KvCacheEvent* event) const {
  auto events = prefix_cache_->get_upload_kvcache_events();
  if (events != nullptr) {
    event->removed_cache.merge(events->removed_cache);
    event->stored_cache.merge(events->stored_cache);
    events->clear();
  }
}

// allocate a block id
Block BlockManagerImpl::allocate() {
  CHECK(num_free_blocks_ > 0) << "No more blocks available";
  size_t prev_count = num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
  const int32_t block_id = free_blocks_[prev_count - 1];
  return {block_id, this};
}

// caller should make sure the block_id is valid
void BlockManagerImpl::free(int32_t block_id) {
  // do nothing for reserved block 0
  if (block_id != 0) {
    size_t prev_count =
        num_free_blocks_.fetch_add(1, std::memory_order_relaxed);
    CHECK(prev_count < free_blocks_.size());
    free_blocks_[prev_count] = block_id;
  }
}

}  // namespace xllm
