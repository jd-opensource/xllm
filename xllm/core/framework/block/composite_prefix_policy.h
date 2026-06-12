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
#include "framework/block/cache_group.h"
#include "framework/kv_cache/kv_cache_tensor_group.h"
#include "util/hash_util.h"
#include "util/slice.h"

namespace xllm {

class KVCacheState;
class PrefixCache;
class PrefixHashState;
class MMData;

// Whether an inserted block came from a full-block insert (incremental groups)
// or a full-restore-chunk insert (DSV4 chunk-atomic groups).
enum class PrefixCacheInsertKind : int8_t {
  FULL_BLOCK = 0,
  FULL_RESTORE_CHUNK = 1,
};

// One block newly inserted into a group-local prefix cache by a flush. Carries
// everything a downstream Hierarchy/store path needs without re-reading the
// cache: the content identity (hash_key/token_end) and the storage location
// (role/device_dp_rank). Only newly-added entries are reported; LRU touches of
// already-present keys are not.
struct PrefixCacheInsertedBlock {
  PrefixCacheGroup group = PrefixCacheGroup::INVALID;
  CacheStorageRole role = CacheStorageRole::DEVICE;
  int32_t device_dp_rank = -1;
  Block block;
  XXH3Key hash_key;
  size_t token_end = 0;
  uint32_t hash_stride = 0;
  PrefixCacheInsertKind insert_kind = PrefixCacheInsertKind::FULL_BLOCK;
};

struct PrefixCacheInsertResult {
  std::vector<PrefixCacheInsertedBlock> inserted_blocks;
};

// Blocks one group contributes to a composite match, in worker export order.
struct CompositeGroupMatch {
  CacheStateId state_id = CacheStateId::C1;
  PrefixCacheGroup group = PrefixCacheGroup::INVALID;
  std::vector<Block> blocks;
};

// The common restorable prefix across all required groups plus, per group, the
// blocks to attach within that prefix.
struct CompositeMatchResult {
  size_t matched_tokens = 0;
  std::vector<CompositeGroupMatch> group_matches;
};

// A group whose blocks participate in prefix caching, paired with its cache.
// The cache is owned by the group's CacheGroupRuntime; the policy only borrows.
struct CacheablePrefixEntry {
  CacheStateId state_id = CacheStateId::C1;
  PrefixCacheGroup group = PrefixCacheGroup::INVALID;
  uint32_t block_size = 0;
  PrefixCache* cache = nullptr;
};

// Inputs the composite resolves from a BlockManagerContext before calling the
// policy. Kept explicit (rather than reaching back through Sequence) so the
// policy is exercisable in isolation.
struct PrefixCacheMatchContext {
  KVCacheState* kv_state = nullptr;
  Slice<int32_t> tokens;
  const MMData* mm_data = nullptr;
  CacheStorageRole role = CacheStorageRole::DEVICE;
  int32_t device_dp_rank = -1;
};

struct PrefixCacheInsertContext {
  KVCacheState* kv_state = nullptr;
  Slice<int32_t> tokens;
  size_t committed_tokens = 0;
  PrefixHashState* hash_state = nullptr;
  const MMData* mm_data = nullptr;
  CacheStorageRole role = CacheStorageRole::DEVICE;
  int32_t device_dp_rank = -1;
};

// Aggregates per-group prefix caches into a single composite match/flush. There
// is exactly one policy per CompositeBlockManager; it owns the cross-group
// alignment that the per-group PrefixCache deliberately does not understand.
class ICompositePrefixPolicy {
 public:
  virtual ~ICompositePrefixPolicy() = default;

  // Matches the prompt prefix, writing each group's shared blocks into the
  // corresponding CacheGroupState and returning the common restorable length.
  virtual CompositeMatchResult match(
      const PrefixCacheMatchContext& context) = 0;

  // Inserts newly-completed full blocks of cacheable groups into their caches,
  // advancing each group's prefix_cached_tokens.
  virtual PrefixCacheInsertResult insert_committed(
      const PrefixCacheInsertContext& context) = 0;
};

// Disables prefix caching: match is always 0, insert never adds anything. Used
// by Qwen3.5+ (Linear state is unrestorable) and the first-phase DSV4 path.
class NoPrefixPolicy final : public ICompositePrefixPolicy {
 public:
  CompositeMatchResult match(const PrefixCacheMatchContext& context) override;
  PrefixCacheInsertResult insert_committed(
      const PrefixCacheInsertContext& context) override;
};

// Normal-model policy: a single C1 prefix cache. The last continuous prefix
// key hit is immediately restorable, and every newly-completed full C1 block is
// inserted in prefill and decode.
class IncrementalOnlyPrefixPolicy final : public ICompositePrefixPolicy {
 public:
  explicit IncrementalOnlyPrefixPolicy(const CacheablePrefixEntry& c1_entry);

  CompositeMatchResult match(const PrefixCacheMatchContext& context) override;
  PrefixCacheInsertResult insert_committed(
      const PrefixCacheInsertContext& context) override;

 private:
  CacheablePrefixEntry c1_;
};

}  // namespace xllm
