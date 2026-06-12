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

#pragma once

#include <cstdint>
#include <vector>

#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache_tensor_group.h"

namespace xllm {

// How a cache state grows blocks over a sequence lifetime.
enum class CachePolicyType : int8_t {
  INCREMENTAL_APPEND = 0,
  ROLLING_WINDOW = 1,
  PER_SEQUENCE_ONCE = 2,
};

// Identity of one cache state inside a group-composite block manager. Unique
// entry key: PrefixCacheGroup cannot be the key because all non-cacheable
// states share PrefixCacheGroup::INVALID.
enum class CacheStateId : int8_t {
  C1 = 0,
  C4 = 1,
  C128 = 2,
  SWA = 3,
  // Generic per-sequence resource block (currently Sequence::single_block_),
  // shared by linear_state_ids / embedding_ids / MTP.
  SINGLE_RES = 4,
};

enum class CacheStorageRole : int8_t {
  DEVICE = 0,
  HOST = 1,
};

// Which worker-side input a cache state's block list is exported to.
enum class WorkerExportTarget : int8_t {
  NONE = 0,
  BLOCK_TABLES = 1,
  MULTI_BLOCK_TABLES = 2,
  LINEAR_STATE_IDS = 3,
  EMBEDDING_IDS = 4,
};

constexpr const char* to_string(CachePolicyType type) {
  switch (type) {
    case CachePolicyType::INCREMENTAL_APPEND:
      return "incremental_append";
    case CachePolicyType::ROLLING_WINDOW:
      return "rolling_window";
    case CachePolicyType::PER_SEQUENCE_ONCE:
      return "per_sequence_once";
  }
  return "unknown";
}

constexpr const char* to_string(CacheStateId state_id) {
  switch (state_id) {
    case CacheStateId::C1:
      return "c1";
    case CacheStateId::C4:
      return "c4";
    case CacheStateId::C128:
      return "c128";
    case CacheStateId::SWA:
      return "swa";
    case CacheStateId::SINGLE_RES:
      return "single_res";
  }
  return "unknown";
}

constexpr const char* to_string(CacheStorageRole role) {
  switch (role) {
    case CacheStorageRole::DEVICE:
      return "device";
    case CacheStorageRole::HOST:
      return "host";
  }
  return "unknown";
}

constexpr const char* to_string(WorkerExportTarget target) {
  switch (target) {
    case WorkerExportTarget::NONE:
      return "none";
    case WorkerExportTarget::BLOCK_TABLES:
      return "block_tables";
    case WorkerExportTarget::MULTI_BLOCK_TABLES:
      return "multi_block_tables";
    case WorkerExportTarget::LINEAR_STATE_IDS:
      return "linear_state_ids";
    case WorkerExportTarget::EMBEDDING_IDS:
      return "embedding_ids";
  }
  return "unknown";
}

// Static configuration of one cache state, fixed at manager construction.
struct CacheGroupSpec {
  CacheStateId state_id = CacheStateId::C1;
  CachePolicyType policy_type = CachePolicyType::INCREMENTAL_APPEND;
  // Only meaningful when prefix_cacheable; non-cacheable states keep INVALID.
  PrefixCacheGroup prefix_group = PrefixCacheGroup::INVALID;
  // Tokens covered by one block of this state.
  uint32_t block_size = 0;
  // Allocator pool size for this state.
  uint32_t num_blocks = 0;
  uint32_t compress_ratio = 1;
  // Ring capacity for ROLLING_WINDOW; 0 for other policies.
  uint32_t window_blocks = 0;
  bool prefix_cacheable = false;
  // SINGLE_RES-like states export to several worker inputs, hence a list.
  std::vector<WorkerExportTarget> export_targets;
  int32_t export_index = -1;
};

// One ring-slot replacement produced by RollingWindowPolicy. The old block is
// retained until the next allocate on the same state commits it (its KV may
// still be inside the attention window if the triggering allocate is rolled
// back).
struct RingSlotReplacement {
  uint32_t slot = 0;
  uint64_t logical_block_idx = 0;
  Block old_block;
};

// Per-sequence runtime state of one cache group.
struct CacheGroupState {
  CacheStateId state_id = CacheStateId::C1;
  // Worker multi_block_tables slot for this group, copied from the owning
  // CacheGroupSpec at state materialization. -1 means the group is not exported
  // to multi_block_tables (e.g. the flat C1 view or SINGLE_RES).
  int32_t export_index = -1;
  std::vector<Block> blocks;
  size_t shared_blocks_num = 0;
  size_t prefix_cached_tokens = 0;

  // For rolling policies.
  uint64_t next_logical_block_idx = 0;
  uint32_t ring_capacity = 0;
  uint32_t ring_start = 0;

  // Rollback bookkeeping for the most recent allocate() on this state. Valid
  // only until the next policy call on the same state.
  size_t last_alloc_new_blocks = 0;
  uint64_t last_alloc_prev_logical_block_idx = 0;
  std::vector<RingSlotReplacement> pending_replacements;
};

}  // namespace xllm
