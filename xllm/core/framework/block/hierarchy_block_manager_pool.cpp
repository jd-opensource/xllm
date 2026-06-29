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
  BlockManagerPool::cache(sequence);

  // Release host blocks if any
  auto host_blocks = sequence->host_kv_state().blocks(BlockType::KV);
  if (!host_blocks.empty()) {
    host_block_managers_[dp_rank]->deallocate(host_blocks);
  }

  // Release device blocks via the composite (includes prefix cache flush)
  auto* composite =
      static_cast<CompositeBlockManager*>(block_managers_[dp_rank].get());
  composite->deallocate_for_sequence(sequence);
  sequence->reset();
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
  for (size_t i = hbm_cache_token_num / options_.block_size();
       i <
       max_copy_in_blocks_num + (hbm_cache_token_num / options_.block_size());
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
          : (max_copy_in_blocks_num +
             (hbm_cache_token_num / options_.block_size())) *
                options_.block_size();

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

    for (size_t i = hbm_cache_token_num / options_.block_size();
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
