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

#include <algorithm>
#include <limits>
#include <unordered_map>

#include "block_manager_impl.h"
#include "common/global_flags.h"
#include "concurrent_block_manager_impl.h"
#include "framework/xtensor/page_allocator.h"
#include "framework/xtensor/phy_page_pool.h"
#include "framework/xtensor/xtensor_block_manager_impl.h"

namespace xllm {
namespace {

struct ReleaseEntry {
  const Block* block = nullptr;
  int64_t refs = 0;
};

using BlockReleaseEntries = std::unordered_map<int32_t, ReleaseEntry>;

void add_block_ref(const Block& block, BlockReleaseEntries* entries) {
  CHECK(entries != nullptr);
  if (!block.is_valid()) {
    return;
  }

  ReleaseEntry& entry = (*entries)[block.id()];
  entry.block = &block;
  ++entry.refs;
}

void add_block_refs(const Slice<Block>& blocks, BlockReleaseEntries* entries) {
  for (const Block& block : blocks) {
    add_block_ref(block, entries);
  }
}

int64_t single_group_release_count(
    const BlockManager& manager,
    const std::vector<Sequence*>& release_candidates,
    int32_t dp_rank,
    bool enable_prefix_cache) {
  BlockReleaseEntries entries;
  for (const Sequence* sequence : release_candidates) {
    if (sequence == nullptr || sequence->dp_rank() != dp_rank) {
      continue;
    }
    add_block_refs(sequence->kv_state().kv_blocks(), &entries);
  }

  int64_t count = 0;
  for (const auto& block_entry : entries) {
    const ReleaseEntry& entry = block_entry.second;
    CHECK(entry.block != nullptr);
    const bool cached =
        enable_prefix_cache && manager.is_prefix_cached(*entry.block);
    const int64_t max_ref_count = entry.refs + (cached ? 1 : 0);
    if (static_cast<int64_t>(entry.block->ref_count()) <= max_ref_count) {
      ++count;
    }
  }
  return count;
}

void add_release_estimates(const SequenceBlockAllocator& allocator,
                           int32_t dp_rank,
                           const std::vector<Sequence*>& release_candidates,
                           std::vector<BlockGroupUsage>* groups) {
  CHECK(groups != nullptr);
  for (BlockGroupUsage& group : *groups) {
    for (const Sequence* sequence : release_candidates) {
      if (sequence == nullptr || sequence->dp_rank() != dp_rank) {
        continue;
      }
      const std::vector<BlockGroupUsage> release =
          allocator.estimate_release(sequence);
      for (const BlockGroupUsage& release_group : release) {
        if (release_group.group_id == group.group_id) {
          group.releasable_blocks += release_group.releasable_blocks;
        }
      }
    }
  }
}

bool has_enough_blocks(const std::vector<BlockGroupUsage>& groups) {
  for (const BlockGroupUsage& group : groups) {
    if (group.free_blocks + group.releasable_blocks < group.needed_blocks) {
      return false;
    }
  }
  return true;
}

void add_single_group_release(int64_t releasable_blocks,
                              std::vector<BlockGroupUsage>* groups) {
  CHECK(groups != nullptr);
  for (BlockGroupUsage& group : *groups) {
    if (group.group_id == 0) {
      group.releasable_blocks += releasable_blocks;
    }
  }
}

void add_release_capacity(const SequenceBlockAllocator& allocator,
                          const BlockManager* manager,
                          int32_t dp_rank,
                          bool composite_blocks,
                          bool enable_prefix_cache,
                          const std::vector<Sequence*>& release_candidates,
                          std::vector<BlockGroupUsage>* groups) {
  CHECK(groups != nullptr);
  if (composite_blocks) {
    add_release_estimates(allocator, dp_rank, release_candidates, groups);
    return;
  }

  CHECK(manager != nullptr);
  const int64_t releasable_blocks = single_group_release_count(
      *manager, release_candidates, dp_rank, enable_prefix_cache);
  add_single_group_release(releasable_blocks, groups);
}

}  // namespace

BlockManagerPool::BlockManagerPool(const Options& options, int32_t dp_size)
    : options_(options) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  block_managers_.reserve(dp_size);
  allocators_.reserve(dp_size);
  single_block_managers_.reserve(dp_size);

  BlockManager::Options block_options;
  block_options.num_blocks(options_.num_blocks())
      .block_size(options_.block_size())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .enable_cache_upload(options_.host_num_blocks() > 0
                               ? false
                               : options_.enable_cache_upload());

  for (int32_t i = 0; i < dp_size; ++i) {
    if (options_.composite_block_plan().has_value()) {
      block_managers_.emplace_back(nullptr);
      allocators_.emplace_back(
          std::make_unique<CompositeSequenceBlockAllocator>(
              options_.composite_block_plan().value()));
    } else if (options_.enable_xtensor()) {
      // Use XTensorBlockManagerImpl for xtensor mode
      CHECK_GT(options_.num_layers(), 0)
          << "num_layers must be set when enable_xtensor is true";
      CHECK_GT(options_.slot_size(), 0)
          << "slot_size must be set when enable_xtensor is true";
      size_t page_size = FLAGS_phy_page_granularity_size;
      // In the current implementation, K and V must be the same size, so we
      // divide by 2.
      size_t block_mem_size =
          static_cast<size_t>(options_.block_size()) * options_.slot_size() / 2;
      block_managers_.emplace_back(
          std::make_unique<XTensorBlockManagerImpl>(block_options,
                                                    options_.num_layers(),
                                                    block_mem_size,
                                                    page_size,
                                                    /*dp_rank=*/i,
                                                    options_.model_id()));
    } else if (options.enable_disagg_pd() || options_.enable_kvcache_store()) {
      block_managers_.emplace_back(
          std::make_unique<ConcurrentBlockManagerImpl>(block_options));
    } else {
      block_managers_.emplace_back(
          std::make_unique<BlockManagerImpl>(block_options));
    }
    if (!options_.composite_block_plan().has_value()) {
      allocators_.emplace_back(std::make_unique<SingleGroupBlockAllocator>(
          block_managers_.back().get()));
    }
    // Scheduler-side per-sequence resources share one logical single-block
    // pool. Worker-side embedding and linear-state caches remain physically
    // separate and are addressed via transport fields.
    single_block_managers_.emplace_back(std::make_unique<SingleBlockManager>(
        /*num_blocks=*/FLAGS_max_seqs_per_batch + 2,
        /*resource_name=*/"single block",
        /*exhaustion_message=*/"No more single-block ids available"));
  }
  swap_block_transfer_infos_.clear();
  swap_block_transfer_infos_.resize(allocators_.size());
}

int32_t BlockManagerPool::get_manager_with_max_free_blocks() const {
  if (allocators_.empty()) {
    return 0;
  }

  size_t max_index = 0;
  int64_t max_free = allocators_[0]->stats().bottleneck_free_sequences;

  for (size_t i = 1; i < allocators_.size(); ++i) {
    const int64_t current_free =
        allocators_[i]->stats().bottleneck_free_sequences;
    if (current_free > max_free) {
      max_free = current_free;
      max_index = i;
    }
  }
  return max_index;
}

int32_t BlockManagerPool::get_dp_rank(Sequence* sequence) const {
  CHECK(sequence != nullptr);
  if (sequence->dp_rank() >= 0) {
    return sequence->dp_rank();
  }
  return get_manager_with_max_free_blocks();
}

int32_t BlockManagerPool::select_dp_rank(Sequence* sequence,
                                         size_t target_num_tokens) const {
  CHECK(sequence != nullptr);
  if (sequence->dp_rank() >= 0) {
    return sequence->dp_rank();
  }

  int32_t selected_rank = -1;
  int64_t selected_margin = std::numeric_limits<int64_t>::min();
  for (size_t i = 0; i < allocators_.size(); ++i) {
    const SequenceAllocEstimate estimate =
        allocators_[i]->estimate_allocate(sequence, target_num_tokens);
    if (!estimate.can_allocate) {
      continue;
    }
    if (estimate.bottleneck_free_sequences > selected_margin) {
      selected_margin = estimate.bottleneck_free_sequences;
      selected_rank = static_cast<int32_t>(i);
    }
  }
  if (selected_rank < 0 && !options_.composite_block_plan().has_value()) {
    return get_manager_with_max_free_blocks();
  }
  return selected_rank;
}

bool BlockManagerPool::allocate_single_block(Sequence* sequence,
                                             int32_t dp_rank) {
  CHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), single_block_managers_.size());
  if (sequence->has_single_block_id()) {
    return true;
  }

  auto single_blocks = single_block_managers_[dp_rank]->allocate(1);
  if (single_blocks.empty()) {
    LOG(ERROR) << "Failed to allocate single block!";
    return false;
  }
  sequence->set_single_block(std::move(single_blocks[0]));
  return true;
}

void BlockManagerPool::deallocate_single_block(Sequence* sequence,
                                               int32_t dp_rank) {
  DCHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), single_block_managers_.size());
  auto single_block = sequence->reset_single_block();
  if (!single_block.is_valid()) {
    return;
  }
  single_block_managers_[dp_rank]->deallocate({&single_block, 1});
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
  allocators_[dp_rank]->deallocate_sequence(sequence);
  deallocate_single_block(sequence, dp_rank);
  // release the blocks after prefix cache insertion
  sequence->reset();
}

std::vector<std::vector<BlockTransferInfo>>*
BlockManagerPool::get_swap_block_transfer_infos() {
  return &swap_block_transfer_infos_;
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
  const bool bound_new_rank = sequence->dp_rank() < 0;
  int32_t dp_rank = select_dp_rank(sequence, num_tokens);
  if (dp_rank < 0) {
    return false;
  }
  sequence->set_dp_rank(dp_rank);
  const bool started_empty = sequence->kv_state().num_kv_blocks() == 0;
  const bool needs_single_block = !sequence->has_single_block_id();
  if (needs_single_block && !allocate_single_block(sequence, dp_rank)) {
    if (bound_new_rank) {
      sequence->set_dp_rank(-1);
    }
    return false;
  }

  if (options_.composite_block_plan().has_value()) {
    if (allocators_[dp_rank]->allocate_sequence(sequence, num_tokens)) {
      return true;
    }
    if (needs_single_block) {
      deallocate_single_block(sequence, dp_rank);
    }
    if (bound_new_rank) {
      sequence->set_dp_rank(-1);
    }
    return false;
  }

  // first try to allocate shared blocks
  if (started_empty) {
    BlockManagerPool::allocate_shared(sequence);
  }

  const size_t num_blocks = sequence->kv_state().num_kv_blocks();
  // round up to the nearest block number
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed <= num_blocks) {
    return process_beam_search(sequence, /*need_swap*/ true);
  }
  process_beam_search(sequence);

  const uint32_t num_additional_blocks = num_blocks_needed - num_blocks;

  const auto blocks = block_managers_[dp_rank]->allocate(num_additional_blocks);
  if (blocks.size() != num_additional_blocks) {
    if (started_empty) {
      block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      sequence->reset();
      if (bound_new_rank) {
        sequence->set_dp_rank(-1);
      }
    }
    // LOG(ERROR) << " Fail to allocate " << num_additional_blocks << "
    // blocks.";

    return false;
  }

  sequence->add_kv_blocks(blocks);

  return true;
}

bool BlockManagerPool::allocate(Sequence* sequence,
                                size_t num_tokens,
                                size_t needed_copy_in_blocks_num) {
  LOG(FATAL)
      << "allocate(Sequence* sequence, size_t num_tokens, size_t "
         "needed_copy_in_blocks_num) is not implemented in BlockManagerPool.";
  return false;
}

std::vector<Block> BlockManagerPool::allocate(size_t num_tokens,
                                              int32_t& dp_rank) {
  CHECK(!options_.composite_block_plan().has_value())
      << "Composite block allocation does not support raw block interfaces";
  dp_rank = get_manager_with_max_free_blocks();
  const size_t block_size = options_.block_size();
  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  return block_managers_[dp_rank]->allocate(num_blocks_needed);
}

bool BlockManagerPool::try_allocate(Sequence* sequence) {
  const bool bound_new_rank = sequence->dp_rank() < 0;
  int32_t dp_rank = select_dp_rank(sequence, sequence->num_tokens());
  if (dp_rank < 0) {
    return false;
  }
  sequence->set_dp_rank(dp_rank);
  const bool needs_single_block = !sequence->has_single_block_id();
  if (needs_single_block && !allocate_single_block(sequence, dp_rank)) {
    if (bound_new_rank) {
      sequence->set_dp_rank(-1);
    }
    return false;
  }

  if (options_.composite_block_plan().has_value()) {
    if (!allocators_[dp_rank]->allocate_sequence(sequence,
                                                 sequence->num_tokens())) {
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      if (bound_new_rank) {
        sequence->set_dp_rank(-1);
      }
      return false;
    }
    return true;
  }

  std::vector<Block> shared_blocks;
  size_t shared_num = 0;
  if (options_.enable_prefix_cache()) {
    const auto& existed_shared_blocks = sequence->kv_state().kv_blocks().slice(
        0, sequence->kv_state().shared_kv_blocks_num());
    // If the sequence holds shared_blocks, the hash values of these blocks do
    // not need to be recalculated and can be reused directly.
    shared_blocks = block_managers_[dp_rank]->allocate_shared(
        sequence->tokens(), existed_shared_blocks);

    if (!shared_blocks.empty()) {
      sequence->add_kv_blocks(shared_blocks);
      sequence->kv_state().incr_shared_kv_blocks_num(shared_blocks.size());
      shared_num = shared_blocks.size();
    }
  }

  const size_t block_size = options_.block_size();
  size_t num_tokens = sequence->tokens().size() - shared_num * block_size;

  const size_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;
  if (num_blocks_needed > 0) {
    const auto blocks = block_managers_[dp_rank]->allocate(num_blocks_needed);
    if (blocks.size() != num_blocks_needed) {
      if (needs_single_block) {
        deallocate_single_block(sequence, dp_rank);
      }
      if (shared_num != 0) {
        block_managers_[dp_rank]->deallocate(shared_blocks);
        sequence->reset();
      }
      if (bound_new_rank) {
        sequence->set_dp_rank(-1);
      }
      return false;
    }

    sequence->add_kv_blocks(std::move(blocks));
  }

  sequence->kv_state().incr_kv_cache_tokens_num(sequence->tokens().size());
  return true;
}

bool BlockManagerPool::process_beam_search(Sequence* sequence, bool need_swap) {
  if (!sequence->check_beam_search()) {
    return true;
  }
  CHECK(!options_.composite_block_plan().has_value())
      << "Composite block allocation does not support beam search";

  auto src_blocks = sequence->kv_state().src_blocks();
  if (src_blocks.size() == 0) {
    return true;
  }

  // when sequence need to swap the last block and no new block appended,
  // allocate a new block for this sequence
  if (need_swap && sequence->kv_state().need_swap()) {
    int32_t dp_rank = get_dp_rank(sequence);
    auto new_blocks = block_managers_[dp_rank]->allocate(1);
    if (new_blocks.size() == 0) {
      return false;
    }
    swap_block_transfer_infos_[dp_rank].emplace_back(src_blocks.back().id(),
                                                     new_blocks[0].id());
    sequence->kv_state().process_beam_search(new_blocks[0]);
  } else {
    sequence->kv_state().process_beam_search(std::nullopt);
  }
  return true;
}

void BlockManagerPool::allocate_shared(Sequence* sequence) {
  // only allocate shared blocks for prefill sequences
  if (options_.enable_prefix_cache()) {
    int32_t dp_rank = sequence->dp_rank();
    if (dp_rank < 0) {
      dp_rank = select_dp_rank(sequence, sequence->num_tokens());
      CHECK_GE(dp_rank, 0);
      sequence->set_dp_rank(dp_rank);
    }
    CHECK(allocators_[dp_rank]->capabilities().supports_prefix_cache)
        << "Selected block allocator does not support prefix cache";
    allocators_[dp_rank]->allocate_shared_sequence(sequence);
  }
}

void BlockManagerPool::cache(Sequence* sequence) {
  int32_t dp_rank = get_dp_rank(sequence);
  if (!options_.enable_prefix_cache() && !options_.enable_cache_upload()) {
    return;
  }
  const BlockAllocatorCapabilities caps = allocators_[dp_rank]->capabilities();
  CHECK(!options_.enable_prefix_cache() || caps.supports_prefix_cache)
      << "Selected block allocator does not support prefix cache";
  CHECK(!options_.enable_cache_upload() || caps.supports_cache_upload)
      << "Selected block allocator does not support cache upload";
  allocators_[dp_rank]->cache_sequence(sequence);
}

void BlockManagerPool::get_merged_kvcache_event(KvCacheEvent* event) const {
  for (int32_t i = 0; i < block_managers_.size(); ++i) {
    if (block_managers_[i] == nullptr) {
      continue;
    }
    block_managers_[i]->get_merged_kvcache_event(event);
  }
}

float BlockManagerPool::get_gpu_cache_usage_perc() const {
  float perc = 0.0;
  for (int32_t i = 0; i < allocators_.size(); ++i) {
    perc += static_cast<float>(allocators_[i]->stats().bottleneck_utilization);
  }
  return perc / allocators_.size();
}

uint32_t BlockManagerPool::num_blocks() const { return options_.num_blocks(); }

int32_t BlockManagerPool::block_size() const { return options_.block_size(); }

std::vector<size_t> BlockManagerPool::num_blocks_in_prefix_cache() const {
  std::vector<size_t> num_blocks_in_prefix_cache(allocators_.size());
  if (!options_.enable_prefix_cache()) {
    return num_blocks_in_prefix_cache;
  }
  for (size_t dp_rank = 0; dp_rank < block_managers_.size(); ++dp_rank) {
    if (block_managers_[dp_rank] == nullptr) {
      num_blocks_in_prefix_cache[dp_rank] = 0;
      continue;
    }
    num_blocks_in_prefix_cache[dp_rank] =
        block_managers_[dp_rank]->num_blocks_in_prefix_cache();
  }
  return num_blocks_in_prefix_cache;
}

std::vector<size_t> BlockManagerPool::num_free_blocks() const {
  std::vector<size_t> num_free_blocks(allocators_.size());
  for (size_t dp_rank = 0; dp_rank < allocators_.size(); ++dp_rank) {
    num_free_blocks[dp_rank] = static_cast<size_t>(
        allocators_[dp_rank]->stats().bottleneck_free_sequences);
  }
  return num_free_blocks;
}

std::vector<size_t> BlockManagerPool::num_used_blocks() const {
  std::vector<size_t> num_used_blocks(allocators_.size());
  for (size_t dp_rank = 0; dp_rank < allocators_.size(); ++dp_rank) {
    const BlockAllocatorStats stats = allocators_[dp_rank]->stats();
    int64_t used_blocks = 0;
    for (const BlockGroupUsage& group : stats.groups) {
      used_blocks = std::max(used_blocks, group.used_blocks);
    }
    num_used_blocks[dp_rank] = static_cast<size_t>(used_blocks);
  }
  return num_used_blocks;
}

double BlockManagerPool::kv_cache_utilization() const {
  int32_t dp_rank = get_manager_with_max_free_blocks();
  return allocators_[dp_rank]->stats().bottleneck_utilization;
}

SequenceAllocEstimate BlockManagerPool::estimate_allocate(
    const Sequence* sequence,
    size_t target_num_tokens) const {
  CHECK(sequence != nullptr);
  if (sequence->dp_rank() >= 0) {
    return allocators_[sequence->dp_rank()]->estimate_allocate(
        sequence, target_num_tokens);
  }

  SequenceAllocEstimate best_estimate;
  int64_t best_margin = std::numeric_limits<int64_t>::min();
  for (const std::unique_ptr<SequenceBlockAllocator>& allocator : allocators_) {
    const SequenceAllocEstimate estimate =
        allocator->estimate_allocate(sequence, target_num_tokens);
    if (!estimate.can_allocate) {
      continue;
    }
    if (estimate.bottleneck_free_sequences > best_margin) {
      best_margin = estimate.bottleneck_free_sequences;
      best_estimate = estimate;
    }
  }
  if (best_margin != std::numeric_limits<int64_t>::min()) {
    return best_estimate;
  }
  return allocators_[get_manager_with_max_free_blocks()]->estimate_allocate(
      sequence, target_num_tokens);
}

std::vector<BlockGroupUsage> BlockManagerPool::estimate_release(
    const Sequence* sequence) const {
  CHECK(sequence != nullptr);
  const int32_t dp_rank = sequence->dp_rank() >= 0
                              ? sequence->dp_rank()
                              : get_manager_with_max_free_blocks();
  return allocators_[dp_rank]->estimate_release(sequence);
}

bool BlockManagerPool::can_allocate_after_release(
    const Sequence* target,
    size_t target_num_tokens,
    const std::vector<Sequence*>& release_candidates) const {
  CHECK(target != nullptr);
  const bool composite_blocks = options_.composite_block_plan().has_value();
  const bool enable_prefix_cache =
      !composite_blocks && options_.enable_prefix_cache();
  if (target->dp_rank() >= 0) {
    const int32_t dp_rank = target->dp_rank();
    SequenceAllocEstimate estimate =
        allocators_[dp_rank]->estimate_allocate(target, target_num_tokens);
    add_release_capacity(*allocators_[dp_rank],
                         block_managers_[dp_rank].get(),
                         dp_rank,
                         composite_blocks,
                         enable_prefix_cache,
                         release_candidates,
                         &estimate.groups);
    return has_enough_blocks(estimate.groups);
  }

  for (size_t dp_rank = 0; dp_rank < allocators_.size(); ++dp_rank) {
    SequenceAllocEstimate estimate =
        allocators_[dp_rank]->estimate_allocate(target, target_num_tokens);
    add_release_capacity(*allocators_[dp_rank],
                         block_managers_[dp_rank].get(),
                         static_cast<int32_t>(dp_rank),
                         composite_blocks,
                         enable_prefix_cache,
                         release_candidates,
                         &estimate.groups);
    if (has_enough_blocks(estimate.groups)) {
      return true;
    }
  }
  return false;
}

// currently use only for profile, which not need prefix cache.
// If more often used in the future, can be integrated into deallocate function.
void BlockManagerPool::deallocate_without_cache(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = get_dp_rank(sequence);
  CHECK(!options_.composite_block_plan().has_value())
      << "Composite block allocation does not support deallocate_without_cache";
  allocators_[dp_rank]->deallocate_sequence(sequence);
  deallocate_single_block(sequence, dp_rank);
  sequence->reset();
}

void BlockManagerPool::reserve_xtensor_padding_blocks() {
  if (!options_.enable_xtensor()) {
    return;
  }
  CHECK(!options_.composite_block_plan().has_value())
      << "Composite block allocation does not support XTensor padding";

  // Reserve padding block on each XTensorBlockManagerImpl.
  for (auto& manager : block_managers_) {
    auto* xtensor_manager =
        dynamic_cast<XTensorBlockManagerImpl*>(manager.get());
    if (xtensor_manager) {
      xtensor_manager->reserve_xtensor_padding_blocks();
    }
  }

  // Start prealloc thread once (PageAllocator is shared by all managers)
  PageAllocator::get_instance().start_prealloc_thread();
}

}  // namespace xllm
