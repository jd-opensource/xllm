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

#include "hierarchy_block_manager_pool.h"

#include "block_manager_impl.h"
#include "concurrent_block_manager_impl.h"

namespace xllm {

HierarchyBlockManagerPool::HierarchyBlockManagerPool(
    const BlockManagerPool::Options& options,
    Engine* engine,
    int32_t dp_size)
    : engine_(engine), BlockManagerPool(options, dp_size) {
  CHECK(dp_size > 0) << "dp_size must be greater than 0";
  CHECK(options_.enable_prefix_cache())
      << "must enable prefix cache for HierarchyBlockManagerPool.";

  if (options_.host_num_blocks() > 0) {
    host_block_managers_.reserve(dp_size);

    BlockManager::Options host_options;
    host_options.num_blocks(options_.num_blocks())
        .block_size(options_.block_size())
        .enable_prefix_cache(options_.enable_prefix_cache())
        .enable_disagg_pd(options_.enable_disagg_pd())
        .num_blocks(options_.host_num_blocks())
        .enable_cache_upload(options_.enable_cache_upload());

    for (int32_t i = 0; i < dp_size; ++i) {
      host_block_managers_.emplace_back(
          std::make_unique<ConcurrentBlockManagerImpl>(host_options));
    }
  }

  load_block_transfer_infos_.resize(block_managers_.size());
  offload_block_pair_queues_.resize(block_managers_.size());
}

void HierarchyBlockManagerPool::deallocate(Sequence* sequence) {
  DCHECK(sequence != nullptr);

  if (!host_block_managers_.empty()) {
    offload_via_host(sequence, true);
  } else {
    offload_direct(sequence, true);
  }
}

void HierarchyBlockManagerPool::offload_via_host(Sequence* sequence,
                                                 bool finish) {
  // add blocks to the prefix cache
  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  const auto token_ids = sequence->cached_tokens();
  BlockManagerPool::cache(sequence);

  auto* blocks = sequence->kv_state().mutable_kv_blocks();
  auto* host_blocks = sequence->host_kv_state().mutable_kv_blocks();

  if (host_blocks->size() > blocks->size()) {
    host_block_managers_[dp_rank]->deallocate(
        sequence->host_kv_state().kv_blocks());
    block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
    sequence->reset();
    return;
  }

  size_t cached_host_block_num =
      sequence->host_kv_state().shared_kv_blocks_num();
  size_t cached_device_block_num = token_ids.size() / options_.block_size();

  size_t needed_block_num = cached_device_block_num > host_blocks->size()
                                ? cached_device_block_num - host_blocks->size()
                                : 0;
  uint32_t needed_offload_num = cached_device_block_num - cached_host_block_num;

  if (needed_offload_num < sequence->get_offload_batch() && !finish) {
    return;
  }

  // allocate additional host blocks for copy
  if (needed_block_num != 0) {
    sequence->host_kv_state().add_kv_blocks(
        host_block_managers_[dp_rank]->allocate(needed_block_num));
  }

  sequence->host_kv_state().incr_shared_kv_blocks_num(needed_offload_num);

  for (size_t i = cached_host_block_num; i < host_blocks->size(); i++) {
    if (blocks->at(i).ref_count() != 2) {
      continue;
    }

    host_blocks->at(i).set_hash_value(blocks->at(i).get_immutable_hash_value());
    auto block_pair = std::make_shared<OffloadBlockPair>(
        blocks->at(i), std::move(host_blocks->at(i)));
    offload_block_pair_queues_[dp_rank].enqueue(std::move(block_pair));
  }

  if (finish) {
    host_block_managers_[dp_rank]->deallocate(
        sequence->host_kv_state().kv_blocks());

    block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
    // release the blocks after prefix cache insertion
    sequence->reset();
  }
}

void HierarchyBlockManagerPool::offload_direct(Sequence* sequence,
                                               bool finish) {
  // add blocks to the prefix cache
  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  const auto token_ids = sequence->cached_tokens();
  BlockManagerPool::cache(sequence);

  auto* blocks = sequence->kv_state().mutable_kv_blocks();

  if (blocks->size() == 0) {
    return;
  }

  size_t cached_block_num = sequence->kv_state().shared_kv_blocks_num();

  size_t full_block_num = token_ids.size() / options_.block_size();

  uint32_t need_offload_num = full_block_num - cached_block_num;

  if (need_offload_num < sequence->get_offload_batch() && !finish) {
    return;
  }

  sequence->kv_state().incr_shared_kv_blocks_num(need_offload_num);

  for (size_t i = cached_block_num; i < full_block_num; i++) {
    if (blocks->at(i).ref_count() != 2) {
      continue;
    }
    auto block_pair = std::make_shared<OffloadBlockPair>(blocks->at(i));
    offload_block_pair_queues_[dp_rank].enqueue(std::move(block_pair));
  }

  if (finish) {
    block_managers_[dp_rank]->deallocate(sequence->kv_state().kv_blocks());
    // release the blocks after prefix cache insertion
    sequence->reset();
  }
}

bool HierarchyBlockManagerPool::allocate(Sequence* sequence,
                                         size_t num_tokens,
                                         size_t max_copy_in_blocks_num) {
  // set needed_kv_cache_tokens_num to overlap computation and data transfer
  if (!BlockManagerPool::allocate(sequence, num_tokens)) {
    return false;
  }

  if (sequence->host_kv_state().num_kv_blocks() == 0 &&
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
    // not enough blocks to copy, return false
    LOG(ERROR) << "Not enough host blocks to copy, max_copy_in_blocks_num: "
               << max_copy_in_blocks_num
               << ", max_copy_blocks_num: " << max_copy_in_blocks_num;
    max_copy_in_blocks_num = max_can_copy_blocks_num;
  }
  auto hbm_blocks = sequence->kv_state().kv_blocks();
  auto host_blocks = sequence->host_kv_state().kv_blocks();
  for (int i = hbm_cache_token_num / options_.block_size();
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
  auto stage = sequence->stage();
  if (!BlockManagerPool::allocate(sequence, num_tokens)) {
    return false;
  }

  switch (stage) {
    case SequenceStage::PREFILL:
    case SequenceStage::CHUNKED_PREFILL:
      if (!host_block_managers_.empty()) {
        load_via_host(sequence, num_tokens);
      } else {
        load_direct(sequence, num_tokens);
      }
      break;
    case SequenceStage::DECODE:
      if (!host_block_managers_.empty()) {
        offload_via_host(sequence, false);
      } else {
        offload_direct(sequence, false);
      }
      break;
    default:
      break;
  }

  return true;
}

void HierarchyBlockManagerPool::load_via_host(Sequence* sequence,
                                              size_t num_tokens) {
  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  allocate_host_shared(sequence);

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
}

void HierarchyBlockManagerPool::load_direct(Sequence* sequence,
                                            size_t num_tokens) {
  int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
  size_t shared_num = sequence->kv_state().shared_kv_blocks_num();
  auto blocks = sequence->kv_state().mutable_kv_blocks();
  const auto& store_matched_blocks = sequence->host_kv_state().kv_blocks();
  uint32_t store_shared_num = 0;
  if (!store_matched_blocks.empty()) {
    // only for chunked prefill, shared blocks in store are already checked in
    // allocate_shared -> allocate_direct_shared, and stored in
    // sequence->host_kv_state().kv_blocks()
    if (store_matched_blocks.size() > shared_num) {
      store_shared_num = store_matched_blocks.size() - shared_num;
      for (int i = shared_num; i < store_matched_blocks.size(); i++) {
        blocks->at(i).set_hash_value(
            store_matched_blocks[i].get_immutable_hash_value());
      }
    } else {
      return;
    }
  } else {
    uint32_t unshared_count =
        PrefixCache::compute_hash_keys(sequence->tokens(), *blocks, shared_num);

    std::vector<uint8_t*> keys;
    keys.reserve(unshared_count);
    for (int i = shared_num; i < blocks->size() - 1; i++) {
      keys.emplace_back(blocks->at(i).get_mutable_hash_value());
    }

    if (keys.empty()) {
      return;
    }

    store_shared_num = PrefixCache::match_in_kvcache_store(keys);
  }

  if (store_shared_num > 0) {
    for (int i = shared_num; i < shared_num + store_shared_num; i++) {
      load_block_transfer_infos_[dp_rank].emplace_back(
          BlockTransferInfo(-1,
                            blocks->at(i).id(),
                            blocks->at(i).get_immutable_hash_value(),
                            TransferType::G2D));
    }

    sequence->kv_state().incr_kv_cache_tokens_num(store_shared_num *
                                                  options_.block_size());
    BlockManagerPool::cache(sequence);
    sequence->kv_state().incr_shared_kv_blocks_num(store_shared_num);
  }
}

void HierarchyBlockManagerPool::allocate_shared(Sequence* sequence) {
  BlockManagerPool::allocate_shared(sequence);
  if (!host_block_managers_.empty()) {
    allocate_host_shared(sequence);
  } else {
    allocate_direct_shared(sequence);
  }
}

void HierarchyBlockManagerPool::allocate_host_shared(Sequence* sequence) {
  if (sequence->host_kv_state().num_kv_blocks() == 0 &&
      sequence->stage() != SequenceStage::DECODE) {
    int32_t dp_rank = BlockManagerPool::get_dp_rank(sequence);
    std::vector<Block> shared_blocks =
        host_block_managers_[dp_rank]->allocate_shared(sequence->tokens());
    sequence->add_shared_host_kv_blocks(std::move(shared_blocks));
  }
}

void HierarchyBlockManagerPool::allocate_direct_shared(Sequence* sequence) {
  if (sequence->host_kv_state().num_kv_blocks() == 0 &&
      sequence->stage() != SequenceStage::DECODE) {
    size_t token_size = sequence->tokens().size();
    size_t block_num =
        (token_size + options_.block_size() - 1) / options_.block_size() - 1;
    auto shared_blocks_num = sequence->kv_state().shared_kv_blocks_num();
    if (shared_blocks_num + 1 >= block_num) {
      return;
    }
    std::vector<Block> blocks(block_num, Block(options_.block_size()));

    uint32_t unshared_count =
        PrefixCache::compute_hash_keys(sequence->tokens(), blocks, 0);

    std::vector<uint8_t*> keys;
    keys.reserve(unshared_count);
    for (int i = 0; i < block_num; i++) {
      keys.emplace_back(blocks[i].get_mutable_hash_value());
    }

    if (keys.empty()) {
      return;
    }

    uint32_t store_shared_num = PrefixCache::match_in_kvcache_store(keys);
    if (store_shared_num > 0) {
      blocks.resize(store_shared_num);
      // using host blocks to record the shared blocks in store
      sequence->host_kv_state().add_kv_blocks(std::move(blocks));
      sequence->host_kv_state().incr_shared_kv_blocks_num(store_shared_num);
      sequence->host_kv_state().incr_kv_cache_tokens_num(store_shared_num *
                                                         options_.block_size());
    }
  }
}

void HierarchyBlockManagerPool::prefetch_from_storage(
    std::shared_ptr<Request>& request) {
  if (!options_.enable_kvcache_store() || host_block_managers_.empty()) {
    return;
  }

  for (auto& prefill_sequence : request->sequences()) {
    DCHECK(prefill_sequence.get() != nullptr);

    int32_t dp_rank = BlockManagerPool::get_dp_rank(prefill_sequence.get());
    std::vector<Block> shared_blocks =
        host_block_managers_[dp_rank]->allocate_shared(
            prefill_sequence->tokens());
    prefill_sequence->add_shared_host_kv_blocks(std::move(shared_blocks));

    // round down to the nearest block number
    size_t shared_blocks_num =
        prefill_sequence->host_kv_state().shared_kv_blocks_num();
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
    prefill_sequence->host_kv_state().add_kv_blocks(host_blocks);
    PrefixCache::compute_hash_keys(
        prefill_sequence->tokens(),
        *prefill_sequence->host_kv_state().mutable_kv_blocks(),
        shared_blocks_num);

    if (num_additional_blocks > 1) {
      const auto host_blocks = prefill_sequence->host_kv_state().kv_blocks();
      std::vector<BlockTransferInfo> block_transfer_infos;
      block_transfer_infos.reserve(num_additional_blocks);
      for (int i = 0; i < num_additional_blocks - 1; i++) {
        block_transfer_infos.emplace_back(BlockTransferInfo(
            -1,
            host_blocks[shared_blocks_num + i].id(),
            host_blocks[shared_blocks_num + i].get_immutable_hash_value(),
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
      auto host_blocks = prefill_sequence->host_kv_state().kv_blocks();
      auto cached_blocks =
          prefill_sequence->host_kv_state().shared_kv_blocks_num();

      host_block_managers_[dp_rank]->cache(
          host_blocks.slice(cached_blocks - success_cnt, cached_blocks));
    }
  }

  return prefetch_result;
}

void HierarchyBlockManagerPool::transfer_blocks(std::vector<Batch>& batches) {
  // load blocks from host to device
  for (size_t i = 0; i < batches.size(); i++) {
    if (!load_block_transfer_infos_[i].empty()) {
      batches[i].set_batch_id();
      engine_->transfer_kv_blocks(
          i, batches[i].batch_id(), std::move(load_block_transfer_infos_[i]));
    }
  }

  load_block_transfer_infos_.clear();
  load_block_transfer_infos_.resize(block_managers_.size());

  transfer_blocks();
}

void HierarchyBlockManagerPool::transfer_blocks() {
  for (int i = 0; i < offload_block_pair_queues_.size(); i++) {
    std::vector<BlockTransferInfo> transfer_infos;
    std::vector<Block> src_blocks;
    std::vector<Block> dst_blocks;

    if (!host_block_managers_.empty()) {
      std::shared_ptr<OffloadBlockPair> block_pair;
      while (offload_block_pair_queues_[i].try_dequeue(block_pair)) {
        src_blocks.emplace_back(std::move(block_pair->src));
        dst_blocks.emplace_back(std::move(block_pair->dst));
        transfer_infos.emplace_back(
            BlockTransferInfo(src_blocks.back().id(),
                              dst_blocks.back().id(),
                              src_blocks.back().get_immutable_hash_value(),
                              TransferType::D2H2G));
        block_pair.reset();
      }

      if (!transfer_infos.empty()) {
        folly::collectAll(std::move(engine_->transfer_kv_blocks(
                              i, std::move(transfer_infos))))
            .via(folly::getGlobalCPUExecutor())
            .thenValue(
                [device_blocks = std::move(src_blocks),
                 host_blocks = std::move(dst_blocks),
                 device_block_mgr_ptr = block_managers_[i].get(),
                 host_block_mgr_ptr = host_block_managers_[i].get()](
                    std::vector<folly::Try<uint32_t>>&& results) mutable {
                  bool offload_success = true;
                  for (auto&& result : results) {
                    if (result.value() != host_blocks.size()) {
                      LOG(ERROR)
                          << "Offload copy fail, expected "
                          << host_blocks.size() << ", got " << result.value();
                      offload_success = false;
                    }
                  }

                  device_block_mgr_ptr->deallocate({device_blocks});
                  device_blocks.clear();

                  if (offload_success) {
                    host_block_mgr_ptr->cache(host_blocks);
                  }
                  host_block_mgr_ptr->deallocate({host_blocks});
                  host_blocks.clear();

                  return 0;
                });
      }
    } else {
      std::shared_ptr<OffloadBlockPair> block_pair;
      while (offload_block_pair_queues_[i].try_dequeue(block_pair)) {
        src_blocks.emplace_back(std::move(block_pair->src));
        transfer_infos.emplace_back(
            BlockTransferInfo(src_blocks.back().id(),
                              -1,
                              src_blocks.back().get_immutable_hash_value(),
                              TransferType::D2G));
        block_pair.reset();
      }

      if (!transfer_infos.empty()) {
        folly::collectAll(std::move(engine_->transfer_kv_blocks(
                              i, std::move(transfer_infos))))
            .via(folly::getGlobalCPUExecutor())
            .thenValue(
                [device_blocks = std::move(src_blocks),
                 device_block_mgr_ptr = block_managers_[i].get()](
                    std::vector<folly::Try<uint32_t>>&& results) mutable {
                  for (auto&& result : results) {
                    if (result.value() != device_blocks.size()) {
                      LOG(ERROR)
                          << "Offload direct to store fail, expected "
                          << device_blocks.size() << ", got " << result.value();
                    }
                  }

                  device_block_mgr_ptr->deallocate({device_blocks});
                  device_blocks.clear();
                  return 0;
                });
      }
    }
  }
}

void HierarchyBlockManagerPool::get_merged_kvcache_event(
    KvCacheEvent* event) const {
  if (host_block_managers_.empty()) {
    BlockManagerPool::get_merged_kvcache_event(event);
  } else {
    for (int32_t i = 0; i < host_block_managers_.size(); ++i) {
      host_block_managers_[i]->get_merged_kvcache_event(event);
    }
  }
}

}  // namespace xllm
