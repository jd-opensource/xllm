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

#include "cache_group_policy.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "framework/block/block_manager.h"

namespace xllm {
namespace {

uint64_t ceil_div(uint64_t numerator, uint64_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

void release_state_blocks(BlockManager* allocator, CacheGroupState* state) {
  std::vector<Block> pending_blocks;
  pending_blocks.reserve(state->pending_replacements.size());
  for (auto& replacement : state->pending_replacements) {
    pending_blocks.emplace_back(std::move(replacement.old_block));
  }
  allocator->deallocate(pending_blocks);
  allocator->deallocate(state->blocks);

  state->blocks.clear();
  state->pending_replacements.clear();
  state->shared_blocks_num = 0;
  state->prefix_cached_tokens = 0;
  state->next_logical_block_idx = 0;
  state->ring_capacity = 0;
  state->ring_start = 0;
  state->last_alloc_new_blocks = 0;
  state->last_alloc_prev_logical_block_idx = 0;
}

}  // namespace

IncrementalAppendPolicy::IncrementalAppendPolicy(const CacheGroupSpec& spec,
                                                 BlockManager* allocator)
    : spec_(spec), allocator_(allocator) {
  CHECK(allocator_ != nullptr);
  CHECK(spec_.policy_type == CachePolicyType::INCREMENTAL_APPEND)
      << "unexpected policy type: " << to_string(spec_.policy_type);
  CHECK_GT(spec_.block_size, 0u);
}

bool IncrementalAppendPolicy::allocate(BlockManagerContext* context,
                                       CacheGroupState* state,
                                       size_t num_tokens) {
  CHECK(state != nullptr);
  state->last_alloc_new_blocks = 0;

  const uint64_t needed_blocks = ceil_div(num_tokens, spec_.block_size);
  if (needed_blocks <= state->blocks.size()) {
    return true;
  }

  const size_t num_new_blocks = needed_blocks - state->blocks.size();
  std::vector<Block> new_blocks = allocator_->allocate(num_new_blocks);
  if (new_blocks.empty()) {
    return false;
  }

  state->blocks.reserve(needed_blocks);
  for (auto& block : new_blocks) {
    state->blocks.emplace_back(std::move(block));
  }
  state->last_alloc_new_blocks = num_new_blocks;
  return true;
}

void IncrementalAppendPolicy::deallocate(BlockManagerContext* context,
                                         CacheGroupState* state) {
  CHECK(state != nullptr);
  release_state_blocks(allocator_, state);
}

void IncrementalAppendPolicy::rollback(BlockManagerContext* context,
                                       CacheGroupState* state) {
  CHECK(state != nullptr);
  CHECK_LE(state->last_alloc_new_blocks, state->blocks.size());
  state->blocks.resize(state->blocks.size() - state->last_alloc_new_blocks);
  state->last_alloc_new_blocks = 0;
}

size_t IncrementalAppendPolicy::additional_blocks_needed(
    const CacheGroupState& state,
    size_t num_tokens) const {
  const uint64_t needed_blocks = ceil_div(num_tokens, spec_.block_size);
  return needed_blocks > state.blocks.size()
             ? needed_blocks - state.blocks.size()
             : 0;
}

RollingWindowPolicy::RollingWindowPolicy(const CacheGroupSpec& spec,
                                         BlockManager* allocator)
    : spec_(spec), allocator_(allocator) {
  CHECK(allocator_ != nullptr);
  CHECK(spec_.policy_type == CachePolicyType::ROLLING_WINDOW)
      << "unexpected policy type: " << to_string(spec_.policy_type);
  CHECK_GT(spec_.block_size, 0u);
  CHECK_GT(spec_.window_blocks, 0u);
}

bool RollingWindowPolicy::allocate(BlockManagerContext* context,
                                   CacheGroupState* state,
                                   size_t num_tokens) {
  CHECK(state != nullptr);
  // The previous allocate on this state is committed by now: the blocks it
  // replaced out are permanently outside the attention window.
  state->pending_replacements.clear();
  state->last_alloc_new_blocks = 0;
  state->last_alloc_prev_logical_block_idx = state->next_logical_block_idx;
  if (num_tokens == 0) {
    return true;
  }

  const uint32_t ring_blocks = spec_.window_blocks;
  if (state->ring_capacity == 0) {
    // admission rule: a new sequence needs the full ring available one-shot
    std::vector<Block> ring = allocator_->allocate(ring_blocks);
    if (ring.empty()) {
      return false;
    }
    state->blocks = std::move(ring);
    state->ring_capacity = ring_blocks;
    state->ring_start = 0;
    state->last_alloc_new_blocks = ring_blocks;
  }
  CHECK_EQ(state->ring_capacity, ring_blocks);
  CHECK_EQ(state->blocks.size(), static_cast<size_t>(ring_blocks));

  const uint64_t needed_logical = ceil_div(num_tokens, spec_.block_size);
  uint64_t logical_idx =
      std::max<uint64_t>(state->next_logical_block_idx, ring_blocks);
  if (needed_logical > logical_idx) {
    state->pending_replacements.reserve(needed_logical - logical_idx);
  }
  for (; logical_idx < needed_logical; ++logical_idx) {
    std::vector<Block> new_block = allocator_->allocate(/*num_blocks=*/1);
    if (new_block.empty()) {
      // undo this call's first-shot ring and slot replacements
      rollback(context, state);
      return false;
    }
    const uint32_t slot = static_cast<uint32_t>(logical_idx % ring_blocks);
    state->pending_replacements.emplace_back(
        RingSlotReplacement{slot, logical_idx, std::move(state->blocks[slot])});
    state->blocks[slot] = std::move(new_block[0]);
  }

  state->next_logical_block_idx =
      std::max(state->next_logical_block_idx, needed_logical);
  return true;
}

void RollingWindowPolicy::deallocate(BlockManagerContext* context,
                                     CacheGroupState* state) {
  CHECK(state != nullptr);
  release_state_blocks(allocator_, state);
}

void RollingWindowPolicy::rollback(BlockManagerContext* context,
                                   CacheGroupState* state) {
  CHECK(state != nullptr);
  // restore in reverse order: one allocate can replace the same slot more
  // than once when it crosses over a full window of logical blocks
  for (auto it = state->pending_replacements.rbegin();
       it != state->pending_replacements.rend();
       ++it) {
    CHECK_LT(static_cast<size_t>(it->slot), state->blocks.size());
    state->blocks[it->slot] = std::move(it->old_block);
  }
  state->pending_replacements.clear();
  state->next_logical_block_idx = state->last_alloc_prev_logical_block_idx;

  if (state->last_alloc_new_blocks > 0) {
    // the rolled-back allocate also performed the first-shot ring allocation
    state->blocks.clear();
    state->ring_capacity = 0;
    state->ring_start = 0;
    state->next_logical_block_idx = 0;
  }
  state->last_alloc_new_blocks = 0;
}

PerSequenceOncePolicy::PerSequenceOncePolicy(const CacheGroupSpec& spec,
                                             BlockManager* allocator)
    : spec_(spec), allocator_(allocator) {
  CHECK(allocator_ != nullptr);
  CHECK(spec_.policy_type == CachePolicyType::PER_SEQUENCE_ONCE)
      << "unexpected policy type: " << to_string(spec_.policy_type);
  CHECK(!spec_.prefix_cacheable)
      << "per-sequence-once states never join prefix cache";
}

bool PerSequenceOncePolicy::allocate(BlockManagerContext* context,
                                     CacheGroupState* state,
                                     size_t num_tokens) {
  CHECK(state != nullptr);
  state->last_alloc_new_blocks = 0;
  if (!state->blocks.empty()) {
    return true;
  }

  std::vector<Block> block = allocator_->allocate(/*num_blocks=*/1);
  if (block.empty()) {
    return false;
  }
  state->blocks = std::move(block);
  state->last_alloc_new_blocks = 1;
  return true;
}

void PerSequenceOncePolicy::deallocate(BlockManagerContext* context,
                                       CacheGroupState* state) {
  CHECK(state != nullptr);
  release_state_blocks(allocator_, state);
}

void PerSequenceOncePolicy::rollback(BlockManagerContext* context,
                                     CacheGroupState* state) {
  CHECK(state != nullptr);
  CHECK_LE(state->last_alloc_new_blocks, state->blocks.size());
  state->blocks.resize(state->blocks.size() - state->last_alloc_new_blocks);
  state->last_alloc_new_blocks = 0;
}

std::unique_ptr<ICacheGroupPolicy> create_cache_group_policy(
    const CacheGroupSpec& spec,
    BlockManager* allocator) {
  switch (spec.policy_type) {
    case CachePolicyType::INCREMENTAL_APPEND:
      return std::make_unique<IncrementalAppendPolicy>(spec, allocator);
    case CachePolicyType::ROLLING_WINDOW:
      return std::make_unique<RollingWindowPolicy>(spec, allocator);
    case CachePolicyType::PER_SEQUENCE_ONCE:
      return std::make_unique<PerSequenceOncePolicy>(spec, allocator);
  }
  LOG(FATAL) << "unknown cache policy type: "
             << static_cast<int32_t>(spec.policy_type);
  return nullptr;
}

}  // namespace xllm
