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

#include "block_manager_pool.h"

#include "block_manager_impl.h"
#include "concurrent_block_manager_impl.h"

namespace xllm {

BlockManagerPool::BlockManagerPool(const BlockManager::Options& options,
                                   int32_t dp_size)
    : options_(options) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  block_managers_.reserve(dp_size);
  host_block_managers_.reserve(dp_size);

  BlockManager::Options npu_options;
  npu_options.num_blocks(options_.num_blocks())
      .block_size(options_.block_size())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .prefix_cache_policy(options_.prefix_cache_policy())
      .enable_service_routing(options_.enable_service_routing());

  BlockManager::Options host_options = npu_options;
  host_options.num_blocks(options_.host_num_blocks())
      .enable_prefix_cache(false)
      .enable_service_routing(false);

  for (int32_t i = 0; i < dp_size; ++i) {
    if (options.enable_disagg_pd()) {
      block_managers_.emplace_back(
          std::make_unique<ConcurrentBlockManagerImpl>(npu_options));
      host_block_managers_.emplace_back(
          std::make_unique<ConcurrentBlockManagerImpl>(host_options));
    } else {
      block_managers_.emplace_back(
          std::make_unique<BlockManagerImpl>(npu_options));
      host_block_managers_.emplace_back(
          std::make_unique<BlockManagerImpl>(host_options));
    }
  }
  reset_copy_content();
}

int32_t BlockManagerPool::get_manager_with_max_free_blocks() const {
  if (block_managers_.empty()) {
    return 0;
  }

  size_t max_index = 0;
  size_t max_free = block_managers_[0]->num_free_blocks();

  for (size_t i = 1; i < block_managers_.size(); ++i) {
    const size_t current_free = block_managers_[i]->num_free_blocks();
    if (current_free > max_free) {
      max_free = current_free;
      max_index = i;
    }
  }
  return max_index;
}

int32_t BlockManagerPool::get_dp_rank(Sequence* sequence) const {
  int32_t dp_rank;
  if (sequence->dp_rank() >= 0) {
    dp_rank = sequence->dp_rank();
  } else {
    dp_rank = get_manager_with_max_free_blocks();
    sequence->set_dp_rank(dp_rank);
  }
  return dp_rank;
}

BlockManager* BlockManagerPool::get_block_manager(Sequence* sequence,
                                                  bool is_host) {
  int32_t dp_rank = get_dp_rank(sequence);
  if (is_host) {
    return host_block_managers_[dp_rank].get();
  } else {
    return block_managers_[dp_rank].get();
  }
}

void BlockManagerPool::deallocate(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    deallocate(sequence.get());
  }
}

void BlockManagerPool::deallocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    deallocate(sequence);
  }
}

void BlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  // add blocks to the prefix cache
  cache(sequence);
  int32_t dp_rank = sequence->dp_rank();
  block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
  // release the blocks after prefix cache insertion
  sequence->reset();
}

void BlockManagerPool::copy_in_blocks_for(Request* request) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    copy_in_blocks_for(&sequence);
  }
}

void BlockManagerPool::copy_in_blocks_for(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    copy_in_blocks_for(sequence);
  }
}

void BlockManagerPool::copy_in_blocks_for(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  auto blocks = sequence->blocks();
  auto host_blocks = sequence->host_blocks();
  auto hbm_shared_blocks =
      sequence->num_kv_cache_tokens() / options_.block_size();

  for (int i = hbm_shared_blocks; i < sequence->get_shared_host_block_num();
       i++) {
    copy_in_cache_contents_[sequence->dp_rank()].emplace_back(
        blocks[i].id(),
        host_blocks[i].id(),
        host_blocks[i].get_immutable_hash_value());
  }
}

void BlockManagerPool::copy_out_blocks_for(Request* request,
                                           bool is_preempted) {
  DCHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    copy_out_blocks_for(&sequence, is_preempted);
  }
}

void BlockManagerPool::copy_out_blocks_for(std::vector<Sequence*>& sequences,
                                           bool is_preempted) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    copy_out_blocks_for(sequence, is_preempted);
  }
}

void BlockManagerPool::copy_out_blocks_for(Sequence* sequence,
                                           bool is_preempted) {
  DCHECK(sequence != nullptr);
  cache(sequence);
  int32_t dp_rank = sequence->dp_rank();
  size_t token_num =
      (sequence->blocks().size() - sequence->host_blocks().size() -
       (sequence->num_tokens() % options_.block_size() != 0)) *
      options_.block_size();
  std::vector<Block>* host_blocks_ptr = sequence->mutable_host_blocks();

  if (token_num > 0) {
    sequence->append_host_blocks(
        host_block_managers_[dp_rank]->allocate(token_num));
  }
  if (!is_preempted) {
    evict_host_blocks_.emplace_back(std::move(*host_blocks_ptr));
    host_blocks_ptr = &evict_host_blocks_.back();
  }

  auto blocks = sequence->blocks();

  for (int i = sequence->get_shared_host_block_num();
       i < host_blocks_ptr->size();
       i++) {
    host_blocks_ptr->at(i).set_hash_value(blocks[i].get_immutable_hash_value());
    copy_out_cache_contents_[dp_rank].emplace_back(
        blocks[i].id(),
        host_blocks_ptr->at(i).id(),
        host_blocks_ptr->at(i).get_immutable_hash_value());
  }
  sequence->set_shared_host_block_num(host_blocks_ptr->size());

  block_managers_[dp_rank]->deallocate(sequence);
}

std::vector<std::vector<CacheContent>>*
BlockManagerPool::get_copy_in_content() {
  return &copy_in_cache_contents_;
}

std::vector<std::vector<CacheContent>>*
BlockManagerPool::get_copy_out_content() {
  return &copy_out_cache_contents_;
}

void BlockManagerPool::reset_copy_content() {
  copy_in_cache_contents_.clear();
  copy_in_cache_contents_.resize(host_block_managers_.size());
  copy_out_cache_contents_.clear();
  copy_out_cache_contents_.resize(host_block_managers_.size());
  evict_host_blocks_.clear();
}

bool BlockManagerPool::allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  return allocate(sequence, sequence->num_tokens());
}

bool BlockManagerPool::allocate(std::vector<Sequence*>& sequences) {
  for (auto* sequence : sequences) {
    DCHECK(sequence != nullptr);
    if (!allocate(sequence, sequence->num_tokens())) {
      // should we gurantee the atomicity of the allocation? all or nothing?
      return false;
    }
  }
  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence, size_t num_tokens) {
  AUTO_COUNTER(allocate_blocks_latency_seconds);
  DCHECK(sequence != nullptr);

  // first try to allocate shared blocks
  if (sequence->kv_state().num_kv_blocks() == 0) {
    allocate_shared(sequence);
  }

  const size_t num_blocks = sequence->kv_state().num_kv_blocks();
  // round up to the nearest block number
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed <= num_blocks) {
    return true;
  }
  const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;

  int32_t dp_rank = get_dp_rank(sequence);
  const auto blocks = block_managers_[dp_rank]->allocate(num_additional_blocks);
  if (blocks.size() != num_additional_blocks) {
    // LOG(ERROR) << " Fail to allocate " << num_additional_blocks << "
    // blocks.";
    return false;
  }

  sequence->add_kv_blocks(blocks);
  return true;
}

std::vector<Block> BlockManagerPool::allocate(size_t num_tokens,
                                              int32_t& dp_rank) {
  dp_rank = get_manager_with_max_free_blocks();
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  return block_managers_[dp_rank]->allocate(num_blocks_needed);
}

void BlockManagerPool::allocate_shared(Sequence* sequence) {
  // only allocate shared blocks for prefill sequences
  if (options_.enable_prefix_cache()) {
    int32_t dp_rank = get_dp_rank(sequence);
    const auto& existed_shared_blocks = sequence->kv_state().kv_blocks().slice(
        0, sequence->kv_state().shared_kv_blocks_num());
    // If the sequence holds shared_blocks, the hash values of these blocks do
    // not need to be recalculated and can be reused directly.
    std::vector<Block> shared_blocks =
        block_managers_[dp_rank]->allocate_shared(sequence->tokens(),
                                                  existed_shared_blocks);
    sequence->add_shared_kv_blocks(std::move(shared_blocks));
  }
}

void BlockManagerPool::cache(Sequence* sequence) {
  int32_t dp_rank = sequence->dp_rank();
  const auto token_ids = sequence->cached_tokens();
  const auto blocks = sequence->kv_state().kv_blocks();
  return block_managers_[dp_rank]->cache(token_ids, blocks);
}

void BlockManagerPool::get_merged_kvcache_event(KvCacheEvent* event) const {
  for (int32_t i = 0; i < block_managers_.size(); ++i) {
    block_managers_[i]->get_merged_kvcache_event(event);
  }
}

float BlockManagerPool::get_gpu_cache_usage_perc() const {
  float perc = 0.0;
  for (int32_t i = 0; i < block_managers_.size(); ++i) {
    perc += block_managers_[i]->get_gpu_cache_usage_perc();
  }
  return perc / block_managers_.size();
}

std::vector<size_t> BlockManagerPool::num_blocks_in_prefix_cache() const {
  std::vector<size_t> num_blocks_in_prefix_cache(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_blocks_in_prefix_cache[dp_rank] =
        block_managers_[dp_rank]->num_blocks_in_prefix_cache();
  }
  return num_blocks_in_prefix_cache;
}

std::vector<size_t> BlockManagerPool::num_free_blocks() const {
  std::vector<size_t> num_free_blocks(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_free_blocks[dp_rank] = block_managers_[dp_rank]->num_free_blocks();
  }
  return num_free_blocks;
}

std::vector<size_t> BlockManagerPool::num_used_blocks() const {
  std::vector<size_t> num_used_blocks(block_managers_.size());
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    num_used_blocks[dp_rank] = block_managers_[dp_rank]->num_used_blocks();
  }
  return num_used_blocks;
}

double BlockManagerPool::kv_cache_utilization() const {
  int32_t dp_rank = get_manager_with_max_free_blocks();
  return block_managers_[dp_rank]->kv_cache_utilization();
}

}  // namespace xllm
