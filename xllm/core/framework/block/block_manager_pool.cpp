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

BlockManagerPool::BlockManagerPool(const Options& options, int32_t dp_size)
    : options_(options) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  block_managers_.reserve(dp_size);
  host_block_managers_.reserve(dp_size);

  BlockManager::Options npu_options;
  npu_options.num_blocks(options_.num_blocks())
      .block_size(options_.block_size())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.enable_cache_upload());

  BlockManager::Options host_options = npu_options;
  host_options.num_blocks(options_.host_num_blocks())
      .enable_cache_upload(false);

  for (int32_t i = 0; i < dp_size; ++i) {
    if (options.enable_disagg_pd() || options_.enable_kvcache_store()) {
      block_managers_.emplace_back(
          std::make_unique<ConcurrentBlockManagerImpl>(npu_options));
      if (options_.host_num_blocks() > 0) {
        host_block_managers_.emplace_back(
            std::make_unique<ConcurrentBlockManagerImpl>(host_options));
      }
    } else {
      block_managers_.emplace_back(
          std::make_unique<BlockManagerImpl>(npu_options));
      if (options_.host_num_blocks() > 0) {
        host_block_managers_.emplace_back(
            std::make_unique<BlockManagerImpl>(host_options));
      }
    }
  }
  reset_transfer_infos();
  offload_block_transfer_infos_.resize(block_managers_.size());
  released_host_blocks_.resize(block_managers_.size());
  released_device_blocks_.resize(block_managers_.size());
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
  int32_t dp_rank = get_dp_rank(sequence);
  cache(sequence);
  if (!host_block_managers_.empty()) {
    record_offload_blocks(sequence);
  }
  block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
  // release the blocks after prefix cache insertion
  sequence->reset();
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_swap_block_transfer_infos() {
  return &swap_block_transfer_infos_;
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_offload_block_transfer_infos() {
  return &offload_block_transfer_infos_;
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_load_block_transfer_infos() {
  return &load_block_transfer_infos_;
}

void BlockManagerPool::set_offload_callback(
    std::vector<std::vector<folly::SemiFuture<uint32_t>>>& futures) {
  DCHECK(futures.size() == block_managers_.size());
  for (int i = 0; i < futures.size(); i++) {
    if (futures[i].empty()) {
      continue;
    }
    // TODO(kangmeng): add timeout
    folly::collectAll(std::move(futures[i]))
        .via(folly::getGlobalCPUExecutor())
        .thenValue([host_blocks = std::move(released_host_blocks_[i]),
                    device_blocks = std::move(released_device_blocks_[i]),
                    host_block_mgr_ptr = host_block_managers_[i].get(),
                    device_block_mgr_ptr = block_managers_[i].get()](
                       std::vector<folly::Try<uint32_t>>&& results) {
          for (auto&& result : results) {
            if (result.value() != host_blocks.size()) {
              LOG(FATAL) << "Offload copy fail, expected " << host_blocks.size()
                         << ", got " << result.value();
            }
          }
          host_block_mgr_ptr->cache(host_blocks);
          host_block_mgr_ptr->deallocate({host_blocks});
          device_block_mgr_ptr->deallocate({device_blocks});
          return 0;
        });
  }

  offload_block_transfer_infos_.clear();
  released_host_blocks_.clear();
  released_device_blocks_.clear();
  offload_block_transfer_infos_.resize(block_managers_.size());
  released_host_blocks_.resize(block_managers_.size());
  released_device_blocks_.resize(block_managers_.size());
}

void BlockManagerPool::reset_transfer_infos() {
  swap_block_transfer_infos_.clear();
  swap_block_transfer_infos_.resize(block_managers_.size());
  load_block_transfer_infos_.clear();
  load_block_transfer_infos_.resize(block_managers_.size());
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
    if (sequence->host_kv_state().num_kv_blocks() == 0) {
      allocate_host_shared(sequence);
    }
  }

  const size_t num_blocks = sequence->kv_state().num_kv_blocks();
  // round up to the nearest block number
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed <= num_blocks) {
    process_beam_search(sequence, /*need_swap*/ true);
    return true;
  }
  process_beam_search(sequence);

  const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;

  int32_t dp_rank = get_dp_rank(sequence);
  const auto blocks = block_managers_[dp_rank]->allocate(num_additional_blocks);
  if (blocks.size() != num_additional_blocks) {
    // LOG(ERROR) << " Fail to allocate " << num_additional_blocks << "
    // blocks.";
    return false;
  }

  sequence->add_kv_blocks(blocks);

  size_t hbm_cache_token_num = sequence->kv_state().kv_cache_tokens_num();
  size_t host_cache_token_num = sequence->host_kv_state().kv_cache_tokens_num();
  if (hbm_cache_token_num < host_cache_token_num) {
    auto hbm_blocks = sequence->kv_state().kv_blocks();
    auto host_blocks = sequence->host_kv_state().kv_blocks();

    for (int i = hbm_cache_token_num / options_.block_size();
         i < host_cache_token_num / options_.block_size();
         i++) {
      load_block_transfer_infos_[dp_rank].emplace_back(
          BlockTransferInfo(host_blocks[i].id(),
                            hbm_blocks[i].id(),
                            host_blocks[i].get_immutable_hash_value(),
                            TransferType::H2D));
    }
    sequence->kv_state().incr_kv_cache_tokens_num(host_cache_token_num -
                                                  hbm_cache_token_num);
  }

  return true;
}

std::vector<Block> BlockManagerPool::allocate(size_t num_tokens,
                                              int32_t& dp_rank) {
  dp_rank = get_manager_with_max_free_blocks();
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  return block_managers_[dp_rank]->allocate(num_blocks_needed);
}

void BlockManagerPool::process_beam_search(Sequence* sequence, bool need_swap) {
  if (!sequence->check_beam_search()) {
    return;
  }

  auto src_blocks = sequence->kv_state().src_blocks();
  if (src_blocks.size() == 0) {
    return;
  }

  // when sequence need to swap the last block and no new block appended,
  // allocate a new block for this sequence
  if (need_swap && sequence->kv_state().need_swap()) {
    int32_t dp_rank = get_dp_rank(sequence);
    auto new_blocks = block_managers_[dp_rank]->allocate(1);
    swap_block_transfer_infos_[dp_rank].emplace_back(src_blocks.back().id(),
                                                     new_blocks[0].id());
    sequence->kv_state().process_beam_search(new_blocks);
  } else {
    sequence->kv_state().process_beam_search({});
  }
}

uint32_t BlockManagerPool::pre_allocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);

  if (!options_.enable_kvcache_store() ||
      sequence->kv_state().num_kv_blocks() != 0 ||
      sequence->host_kv_state().num_kv_blocks() != 0) {
    return 0;
  }

  int32_t dp_rank = get_dp_rank(sequence);
  allocate_host_shared(sequence);

  const size_t num_blocks = sequence->host_kv_state().num_kv_blocks();
  // round down to the nearest block number
  const size_t block_size = options_.block_size();
  const size_t num_additional_blocks =
      sequence->num_tokens() / block_size - num_blocks;
  if (num_additional_blocks <= 0) {
    return 0;
  }

  auto host_blocks =
      host_block_managers_[dp_rank]->allocate(num_additional_blocks);
  if (host_blocks.size() != num_additional_blocks) {
    return 0;
  }

  PrefixCache::compute_hash_keys(sequence->tokens(), host_blocks);

  sequence->host_kv_state().add_kv_blocks(host_blocks);
  return num_additional_blocks;
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
  int32_t dp_rank = get_dp_rank(sequence);
  const auto token_ids = sequence->cached_tokens();
  auto* blocks = sequence->kv_state().mutable_kv_blocks();
  block_managers_[dp_rank]->cache(token_ids, *blocks);
}

void BlockManagerPool::allocate_host_shared(Sequence* sequence) {
  // only allocate shared blocks for prefill sequences
  if (sequence->host_kv_state().num_kv_blocks() != 0 ||
      host_block_managers_.size() != block_managers_.size()) {
    return;
  }

  if (options_.enable_prefix_cache()) {
    int32_t dp_rank = get_dp_rank(sequence);
    std::vector<Block> shared_blocks =
        host_block_managers_[dp_rank]->allocate_shared(sequence->tokens());
    sequence->add_shared_host_kv_blocks(std::move(shared_blocks));
  }
}

void BlockManagerPool::record_offload_blocks(Sequence* sequence) {
  DCHECK(sequence != nullptr);

  auto* blocks = sequence->kv_state().mutable_kv_blocks();
  auto* host_blocks = sequence->host_kv_state().mutable_kv_blocks();

  if (blocks->size() == 0 || host_blocks->size() >= blocks->size()) {
    return;
  }

  int cached_block_num =
      sequence->host_kv_state().kv_cache_tokens_num() / options_.block_size();

  int32_t dp_rank = get_dp_rank(sequence);

  if (host_blocks->size() > 0) {
    host_block_managers_[dp_rank]->cache(sequence->tokens(), *host_blocks);
  }

  size_t needed_block_num =
      sequence->num_tokens() / options_.block_size() - host_blocks->size();

  if (needed_block_num == 0) {
    return;
  }

  sequence->host_kv_state().add_kv_blocks(
      host_block_managers_[dp_rank]->allocate(needed_block_num));

  for (int i = cached_block_num; i < host_blocks->size(); i++) {
    if (blocks->at(i).ref_count() != 2) {
      continue;
    }

    host_blocks->at(i).set_hash_value(blocks->at(i).get_immutable_hash_value());
    released_host_blocks_[dp_rank].emplace_back(std::move(host_blocks->at(i)));
    released_device_blocks_[dp_rank].emplace_back(std::move(blocks->at(i)));
    offload_block_transfer_infos_[dp_rank].emplace_back(BlockTransferInfo(
        released_device_blocks_[dp_rank].back().id(),
        released_host_blocks_[dp_rank].back().id(),
        released_host_blocks_[dp_rank].back().get_immutable_hash_value(),
        TransferType::D2G));
  }
  host_block_managers_[dp_rank]->cache(
      *sequence->host_kv_state().mutable_kv_blocks());
  host_block_managers_[dp_rank]->deallocate(
      sequence->host_kv_state().kv_blocks());
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

uint32_t BlockManagerPool::num_blocks() const { return options_.num_blocks(); }

int32_t BlockManagerPool::block_size() const { return options_.block_size(); }

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
