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

#include "linear_state_cache_manager.h"

#include <algorithm>

#include "framework/model/model_input_params.h"
#include "framework/prefix_cache/block_hasher.h"
#include "framework/request/sequence.h"

namespace xllm {

LinearStateCacheManager::LinearStateCacheManager(const Options& options)
    : options_(options) {
  CHECK_GT(options_.num_slots(), 0)
      << "num_slots must be positive when LinearStateCacheManager is created";
  CHECK_GT(options_.block_size(), 0)
      << "block_size must be positive when LinearStateCacheManager is created";
  const int32_t dp_size = options_.dp_size();
  CHECK_GT(dp_size, 0);
  caches_.reserve(dp_size);
  for (int32_t i = 0; i < dp_size; ++i) {
    caches_.emplace_back(
        std::make_unique<LinearStatePrefixCache>(options_.num_slots()));
  }
}

bool LinearStateCacheManager::allocate_slot(Sequence* sequence,
                                            int32_t dp_rank) {
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), caches_.size());
  if (sequence->has_linear_state_slot()) {
    return true;
  }
  Block slot = caches_[dp_rank]->allocate_live_slot();
  if (!slot.is_valid()) {
    LOG(ERROR) << "Failed to acquire linear state slot!";
    return false;
  }
  sequence->set_linear_state_slot(std::move(slot));
  return true;
}

void LinearStateCacheManager::release_slot(Sequence* sequence) {
  sequence->reset_linear_state_slot();
}

LinearStatePrefixCache* LinearStateCacheManager::prefix_cache(int32_t dp_rank) {
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), caches_.size());
  return caches_[dp_rank].get();
}

LinearStateCheckpointReservations LinearStateCacheManager::resolve_cache_ops(
    int32_t dp_rank,
    std::vector<LinearStateCacheOp>* cache_ops,
    const std::vector<Sequence*>& sequences) {
  LinearStatePrefixCache* cache = prefix_cache(dp_rank);
  LinearStateCheckpointReservations checkpoint_reservations;
  if (cache != nullptr && cache_ops != nullptr && !cache_ops->empty()) {
    checkpoint_reservations = cache->resolve_cache_ops(cache_ops, sequences);
  }
  checkpoint_reservations.dp_rank = dp_rank;
  return checkpoint_reservations;
}

void LinearStateCacheManager::commit_reservations(
    LinearStateCheckpointReservations&& reservations) {
  if (reservations.dp_rank < 0) {
    return;
  }
  LinearStatePrefixCache* cache = prefix_cache(reservations.dp_rank);
  if (cache == nullptr) {
    return;
  }
  cache->commit_reservations(std::move(reservations));
}

size_t LinearStateCacheManager::compute_safe_shared_prefix_length(
    int32_t dp_rank,
    const Sequence* sequence,
    size_t existed_shared_blocks_num,
    size_t total_shared_blocks) const {
  CHECK(sequence != nullptr);
  CHECK_GE(dp_rank, 0);
  CHECK_LT(static_cast<size_t>(dp_rank), caches_.size());

  const LinearStatePrefixCache* pool = caches_[dp_rank].get();
  const int32_t block_size = options_.block_size();
  if (block_size <= 0) {
    return total_shared_blocks;
  }
  size_t safe_shared_blocks =
      std::min(existed_shared_blocks_num, total_shared_blocks);
  const size_t max_reusable_blocks =
      sequence->num_tokens() == 0
          ? 0
          : (sequence->num_tokens() - 1) / static_cast<size_t>(block_size);
  const size_t reusable_limit = std::max(
      safe_shared_blocks, std::min(total_shared_blocks, max_reusable_blocks));

  const size_t token_blocks =
      sequence->tokens().size() / static_cast<size_t>(block_size);
  const size_t hash_limit = std::min(reusable_limit, token_blocks);
  const std::vector<PrefixHash> hashes = compute_linear_state_prefix_hashes(
      sequence->tokens(),
      static_cast<size_t>(block_size),
      hash_limit * static_cast<size_t>(block_size));
  for (size_t block_idx = safe_shared_blocks; block_idx < hashes.size();
       ++block_idx) {
    if (pool->contains(XXH3Key(hashes[block_idx].data()))) {
      safe_shared_blocks = block_idx + 1;
    }
  }
  return std::min(safe_shared_blocks, total_shared_blocks);
}

}  // namespace xllm
