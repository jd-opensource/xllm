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

#include "hierarchy_block_manager_pool.h"

#include <algorithm>

#include "block_manager_impl.h"
#include "composite_block_manager.h"
#include "concurrent_block_manager_impl.h"

namespace xllm {

HierarchyBlockManagerPool::HierarchyBlockManagerPool(
    const BlockManagerPool::Options& options,
    Engine* engine,
    int32_t dp_size)
    : engine_(engine), BlockManagerPool(options, dp_size) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  host_block_managers_.reserve(dp_size);

  BlockManager::Options host_options;
  host_options.num_blocks(options_.host_num_blocks())
      .block_size(options_.block_size())
      .enable_prefix_cache(options_.enable_prefix_cache())
      .enable_disagg_pd(options_.enable_disagg_pd())
      .hasher_type(options_.hasher_type());

  for (int32_t i = 0; i < dp_size; ++i) {
    std::unique_ptr<BlockManager> leaf =
        std::make_unique<BlockManagerImpl>(host_options);
    if (options.enable_disagg_pd() || options_.enable_kvcache_store()) {
      leaf = std::make_unique<ConcurrentBlockManagerImpl>(std::move(leaf));
    }
    host_block_managers_.emplace_back(std::move(leaf));
  }

  load_block_transfer_infos_.resize(host_block_managers_.size());
  offload_block_pair_queues_.resize(host_block_managers_.size());
}

void HierarchyBlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);
  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  // Publish device KV blocks into the device prefix cache first, so that
  // offload-eligible blocks reach ref_count == 2 (held only by the sequence
  // state and the prefix-cache node). deallocate_for_sequence below re-runs
  // this cache step; it is idempotent (hashes recomputed from token ids find
  // the nodes inserted here), so moved-out blocks do not corrupt the cache.
  BlockManagerPool::cache(sequence);

  collect_offload_pairs(sequence, dp_rank);

  // Release the host blocks still held by the sequence. Blocks moved into the
  // offload queue are now invalid in this vector and are skipped by deallocate;
  // their host ids stay reserved (held by the queue) until the D2H copy
  // completes and the offload callback caches + frees them.
  auto host_blocks = sequence->host_kv_state().blocks(BlockType::KV);
  if (!host_blocks.empty()) {
    host_block_managers_[dp_rank]->deallocate(host_blocks);
  }

  // Release device blocks via the composite (includes prefix cache flush).
  // Offloaded device blocks were moved out above, so the KV leaf skips them;
  // the offload callback releases them once the copy is done.
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  composite->deallocate_for_sequence(sequence);
  sequence->reset();
}

void HierarchyBlockManagerPool::collect_offload_pairs(Sequence* sequence,
                                                      int32_t dp_rank) {
  if (!options_.enable_prefix_cache()) {
    return;
  }

  std::vector<Block>* device_blocks =
      sequence->kv_state().mutable_blocks(BlockType::KV);
  std::vector<Block>* host_blocks =
      sequence->host_kv_state().mutable_blocks(BlockType::KV);
  if (device_blocks == nullptr || device_blocks->empty()) {
    return;
  }

  const size_t block_size = options_.block_size();
  const size_t cached_host_block_num =
      sequence->host_kv_state().kv_cache_tokens_num() / block_size;
  const size_t cached_device_block_num =
      sequence->kv_state().kv_cache_tokens_num() / block_size;

  const size_t host_block_num =
      host_blocks == nullptr ? 0 : host_blocks->size();
  // Host already holds at least as many blocks as the device computed: nothing
  // new to offload.
  if (host_block_num >= device_blocks->size()) {
    return;
  }

  // Allocate the host blocks needed to receive the device blocks that have no
  // host counterpart yet.
  const size_t needed_block_num = cached_device_block_num > host_block_num
                                      ? cached_device_block_num - host_block_num
                                      : 0;
  if (needed_block_num != 0) {
    std::vector<Block> new_host_blocks =
        host_block_managers_[dp_rank]->allocate(needed_block_num);
    if (new_host_blocks.size() != needed_block_num) {
      // Host pool exhausted; skip offload this round rather than partially
      // copy.
      return;
    }
    sequence->add_host_blocks(BlockType::KV, new_host_blocks);
    host_blocks = sequence->host_kv_state().mutable_blocks(BlockType::KV);
  }
  if (host_blocks == nullptr) {
    return;
  }

  // Only offload blocks that are fully computed on device. In-batch prefix
  // cache insertion may register blocks before they are computed, so bound the
  // offload range by cached_device_block_num to avoid copying uncomputed data.
  const size_t offload_end_block_num = std::min(
      {cached_device_block_num, host_blocks->size(), device_blocks->size()});
  for (size_t i = cached_host_block_num; i < offload_end_block_num; i++) {
    // ref_count == 2 means the block is held only by this sequence and the
    // prefix-cache node, i.e. it is uniquely owned and safe to offload. Beam
    // forks (shared blocks) have ref_count > 2 and are skipped.
    if (device_blocks->at(i).ref_count() != 2) {
      continue;
    }
    host_blocks->at(i).set_hash_value(
        device_blocks->at(i).get_immutable_hash_value());
    auto block_pair = std::make_shared<OffloadBlockPair>(
        std::move(device_blocks->at(i)), std::move(host_blocks->at(i)));
    offload_block_pair_queues_[dp_rank].enqueue(std::move(block_pair));
  }
}

bool HierarchyBlockManagerPool::allocate(Sequence* sequence,
                                         size_t num_tokens,
                                         size_t max_copy_in_blocks_num) {
  if (!BlockManagerPool::allocate(sequence, num_tokens)) {
    return false;
  }

  if (sequence->host_kv_state().num_blocks(BlockType::KV) == 0 &&
      sequence->stage() != SequenceStage::DECODE) {
    allocate_host_shared(sequence);
  }

  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  size_t hbm_cache_token_num = sequence->kv_state().kv_cache_tokens_num();
  size_t host_cache_token_num = sequence->host_kv_state().kv_cache_tokens_num();
  size_t max_can_copy_blocks_num =
      host_cache_token_num > hbm_cache_token_num
          ? host_cache_token_num / options_.block_size() -
                hbm_cache_token_num / options_.block_size()
          : 0;
  if (max_copy_in_blocks_num > max_can_copy_blocks_num) {
    LOG(ERROR) << "Not enough host blocks to copy, max_copy_in_blocks_num: "
               << max_copy_in_blocks_num
               << ", max_copy_blocks_num: " << max_can_copy_blocks_num;
    max_copy_in_blocks_num = max_can_copy_blocks_num;
  }
  auto hbm_blocks = sequence->kv_state().blocks(BlockType::KV);
  auto host_blocks = sequence->host_kv_state().blocks(BlockType::KV);
  // H2D copies host block i -> device block i, so i must index both vectors.
  // The host prefix match (host_cache_token_num) is computed over the full
  // prompt and can exceed the device blocks allocated for this (possibly
  // chunked) num_tokens, so clamp the copy range to the blocks that actually
  // exist on both sides to avoid out-of-bounds reads.
  const size_t hbm_block_begin = hbm_cache_token_num / options_.block_size();
  const size_t copy_block_limit =
      std::min(hbm_blocks.size(), host_blocks.size());
  if (hbm_block_begin + max_copy_in_blocks_num > copy_block_limit) {
    max_copy_in_blocks_num = copy_block_limit > hbm_block_begin
                                 ? copy_block_limit - hbm_block_begin
                                 : 0;
  }
  for (size_t i = hbm_block_begin; i < max_copy_in_blocks_num + hbm_block_begin;
       i++) {
    load_block_transfer_infos_[dp_rank].emplace_back(
        BlockTransferInfo(host_blocks[i].id(),
                          hbm_blocks[i].id(),
                          host_blocks[i].get_immutable_hash_value(),
                          TransferType::H2D));
  }

  size_t target_hbm_cache_token_num =
      max_copy_in_blocks_num == 0
          ? hbm_cache_token_num
          : (max_copy_in_blocks_num + hbm_block_begin) * options_.block_size();

  sequence->kv_state().incr_kv_cache_tokens_num(target_hbm_cache_token_num -
                                                hbm_cache_token_num);

  return true;
}

bool HierarchyBlockManagerPool::allocate(Sequence* sequence,
                                         size_t num_tokens) {
  if (!BlockManagerPool::allocate(sequence, num_tokens)) {
    return false;
  }

  if (sequence->host_kv_state().num_blocks(BlockType::KV) == 0 &&
      sequence->stage() != SequenceStage::DECODE) {
    allocate_host_shared(sequence);
  }

  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  size_t hbm_cache_token_num = sequence->kv_state().kv_cache_tokens_num();
  size_t host_cache_token_num = sequence->host_kv_state().kv_cache_tokens_num();
  if (hbm_cache_token_num < host_cache_token_num) {
    auto hbm_blocks = sequence->kv_state().blocks(BlockType::KV);
    auto host_blocks = sequence->host_kv_state().blocks(BlockType::KV);

    // H2D copies host block i -> device block i. host_cache_token_num is the
    // host prefix match over the full prompt and can exceed the device blocks
    // allocated for this (chunked) num_tokens, so clamp to the blocks present
    // on both sides to avoid out-of-bounds reads on hbm_blocks.
    const size_t copy_block_end =
        std::min({host_cache_token_num / options_.block_size(),
                  hbm_blocks.size(),
                  host_blocks.size()});
    const size_t hbm_block_begin = hbm_cache_token_num / options_.block_size();
    for (size_t i = hbm_block_begin; i < copy_block_end; i++) {
      load_block_transfer_infos_[dp_rank].emplace_back(
          BlockTransferInfo(host_blocks[i].id(),
                            hbm_blocks[i].id(),
                            host_blocks[i].get_immutable_hash_value(),
                            TransferType::H2D));
    }
    const size_t target_hbm_cache_token_num =
        copy_block_end > hbm_block_begin
            ? copy_block_end * options_.block_size()
            : hbm_cache_token_num;
    sequence->kv_state().incr_kv_cache_tokens_num(target_hbm_cache_token_num -
                                                  hbm_cache_token_num);
  }
  return true;
}

void HierarchyBlockManagerPool::allocate_shared(Sequence* sequence) {
  BlockManagerPool::allocate_shared(sequence);
  if (sequence->host_kv_state().num_blocks(BlockType::KV) == 0 &&
      sequence->stage() != SequenceStage::DECODE) {
    allocate_host_shared(sequence);
  }
}

void HierarchyBlockManagerPool::allocate_host_shared(Sequence* sequence) {
  if (options_.enable_prefix_cache()) {
    int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
    std::vector<Block> shared_blocks =
        host_block_managers_[dp_rank]->allocate_shared(sequence->tokens());
    sequence->add_shared_host_blocks(BlockType::KV, std::move(shared_blocks));
  }
}

void HierarchyBlockManagerPool::prefetch_from_storage(
    std::shared_ptr<Request>& request) {
  if (!options_.enable_kvcache_store()) {
    return;
  }

  for (auto& prefill_sequence : request->sequences()) {
    DCHECK(prefill_sequence.get() != nullptr);

    int32_t dp_rank = BlockManagerPool::get_dp_rank(prefill_sequence.get());
    std::vector<Block> shared_blocks =
        host_block_managers_[dp_rank]->allocate_shared(
            prefill_sequence->tokens());
    prefill_sequence->add_shared_host_blocks(BlockType::KV,
                                             std::move(shared_blocks));

    size_t shared_blocks_num =
        prefill_sequence->host_kv_state().shared_blocks_num(BlockType::KV);
    const size_t num_additional_blocks =
        (prefill_sequence->num_tokens() + options_.block_size() - 1) /
            options_.block_size() -
        shared_blocks_num;
    if (num_additional_blocks <= 1) {
      continue;
    }

    auto host_blocks =
        host_block_managers_[dp_rank]->allocate(num_additional_blocks);
    if (host_blocks.size() != num_additional_blocks) {
      continue;
    }
    prefill_sequence->add_host_blocks(BlockType::KV, host_blocks);
    PrefixCache::compute_hash_keys(
        prefill_sequence->tokens(),
        *prefill_sequence->host_kv_state().mutable_blocks(BlockType::KV),
        shared_blocks_num);

    if (num_additional_blocks > 1) {
      const auto host_blks =
          prefill_sequence->host_kv_state().blocks(BlockType::KV);
      std::vector<BlockTransferInfo> block_transfer_infos;
      block_transfer_infos.reserve(num_additional_blocks);
      for (size_t i = 0; i < num_additional_blocks - 1; i++) {
        block_transfer_infos.emplace_back(BlockTransferInfo(
            -1,
            host_blks[shared_blocks_num + i].id(),
            host_blks[shared_blocks_num + i].get_immutable_hash_value(),
            TransferType::G2H));
      }

      engine_->prefetch_from_storage(prefill_sequence->dp_rank(),
                                     std::move(block_transfer_infos),
                                     prefill_sequence->get_termination_flag(),
                                     prefill_sequence->get_prefetch_results());
    }
  }
}

bool HierarchyBlockManagerPool::update_prefetch_result(
    std::shared_ptr<Request>& request,
    const uint32_t timeout) {
  if (!options_.enable_kvcache_store()) {
    return true;
  }

  bool prefetch_result = true;
  for (auto& prefill_sequence : request->sequences()) {
    uint32_t success_cnt = 0;
    prefetch_result &=
        prefill_sequence->update_prefetch_result(timeout, success_cnt);

    if (prefetch_result && success_cnt > 0) {
      int32_t dp_rank = BlockManagerPool::get_dp_rank(prefill_sequence.get());
      auto host_blocks =
          prefill_sequence->host_kv_state().blocks(BlockType::KV);
      auto cached_blocks =
          prefill_sequence->host_kv_state().shared_blocks_num(BlockType::KV);

      host_block_managers_[dp_rank]->cache(
          host_blocks.slice(cached_blocks - success_cnt, cached_blocks));
    }
  }

  return prefetch_result;
}

void HierarchyBlockManagerPool::transfer_blocks(std::vector<Batch>& batches) {
  for (size_t i = 0; i < batches.size(); i++) {
    if (!load_block_transfer_infos_[i].empty()) {
      batches[i].set_batch_id();
      engine_->transfer_kv_blocks(
          i, batches[i].batch_id(), std::move(load_block_transfer_infos_[i]));
    }
  }

  load_block_transfer_infos_.clear();
  load_block_transfer_infos_.resize(host_block_managers_.size());

  transfer_blocks();
}

void HierarchyBlockManagerPool::transfer_blocks() {
  for (size_t i = 0; i < offload_block_pair_queues_.size(); i++) {
    std::vector<BlockTransferInfo> transfer_infos;
    std::vector<Block> src_blocks;
    std::vector<Block> dst_blocks;

    std::shared_ptr<OffloadBlockPair> block_pair;
    while (offload_block_pair_queues_[i].try_dequeue(block_pair)) {
      src_blocks.emplace_back(std::move(block_pair->src));
      dst_blocks.emplace_back(std::move(block_pair->dst));
      transfer_infos.emplace_back(
          BlockTransferInfo(src_blocks.back().id(),
                            dst_blocks.back().id(),
                            dst_blocks.back().get_immutable_hash_value(),
                            TransferType::D2H2G));
      block_pair.reset();
    }

    if (!transfer_infos.empty()) {
      folly::collectAll(
          std::move(engine_->transfer_kv_blocks(i, std::move(transfer_infos))))
          .via(folly::getGlobalCPUExecutor())
          .thenValue([device_blocks = std::move(src_blocks),
                      host_blocks = std::move(dst_blocks),
                      device_block_mgr_ptr = block_managers_[i].get(),
                      host_block_mgr_ptr = host_block_managers_[i].get()](
                         std::vector<folly::Try<uint32_t>>&& results) mutable {
            for (auto&& result : results) {
              if (result.value() != host_blocks.size()) {
                LOG(FATAL) << "Offload copy fail, expected "
                           << host_blocks.size() << ", got " << result.value();
              }
            }

            device_block_mgr_ptr->deallocate({device_blocks});
            device_blocks.clear();

            host_block_mgr_ptr->cache(host_blocks);
            host_block_mgr_ptr->deallocate({host_blocks});
            host_blocks.clear();

            return 0;
          });
    }
  }
}

}  // namespace xllm
