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

#include <cstddef>
#include <memory>
#include <vector>

#include "framework/block/block_manager.h"
#include "framework/block/block_manager_context.h"
#include "framework/block/cache_group.h"
#include "framework/block/cache_group_policy.h"
#include "framework/block/composite_prefix_policy.h"
#include "util/slice.h"

namespace xllm {

class KVCacheState;
class PrefixCache;
class PrefixHashState;
class MMData;

// One cache state's owned allocation machinery inside a composite manager: its
// static spec, the leaf allocator backing its block pool, the policy that grows
// and shrinks it, and (only for prefix-cacheable groups) the group-local prefix
// cache. The cache's hash stride equals the group's block_size by construction.
// Move-only because it owns the allocator, policy, and cache.
struct CacheGroupRuntime {
  CacheGroupSpec spec;
  std::unique_ptr<BlockManager> allocator;
  std::unique_ptr<ICacheGroupPolicy> policy;
  // Null for non-cacheable groups; owned by the runtime, borrowed by the
  // composite prefix policy.
  std::unique_ptr<PrefixCache> prefix_cache;
};

// Sequence-level orchestrator over several cache states (C1 / DSV4 C4 / C128,
// SWA, SINGLE_RES). Composition, not inheritance: it does not implement the
// BlockManager block-pool interface. Instead it drives one ICacheGroupPolicy
// per group against the per-sequence CacheGroupState vector carried on the
// sequence's KVCacheState.
//
// allocate() is all-or-nothing across groups: if any group cannot grow, the
// groups already grown in the same call are rolled back so the sequence's
// block tables stay consistent. Prefix-cache integration is deferred to a
// later migration step, so the leaf allocators are built with prefix cache
// disabled and act as pure block pools.
//
// This class is NOT internally synchronized: the bare orchestrator is used on
// the single-threaded scheduler path. Modes with concurrent sequence-level
// entry (disagg PD) use the ConcurrentCompositeBlockManager subclass, which
// overrides the sequence-level entries with a lock. The leaf allocators are
// independently thread-safe, so the free path (Block destructors ->
// allocator->free()) never needs this layer.
class GroupCompositeBlockManager {
 public:
  explicit GroupCompositeBlockManager(const std::vector<CacheGroupSpec>& specs);
  // Out-of-line: the unique_ptr members hold forward-declared types whose full
  // definitions are only visible in the translation unit.
  virtual ~GroupCompositeBlockManager();

  // Grow every group to cover `num_tokens` committed tokens for the sequence
  // bound by `context`. Returns false (and leaves all groups unchanged) when
  // any single group cannot satisfy the request. Concurrent calls must be
  // serialized by the caller or by the concurrent subclass.
  virtual bool allocate(BlockManagerContext* context, size_t num_tokens);

  // Release every group held by the sequence bound by `context`.
  virtual void deallocate(BlockManagerContext* context);

  // Match `tokens` against the group-local prefix caches, attaching each
  // group's shared blocks to the sequence's per-group state and returning the
  // common restorable prefix. Must run on the sequence's FIRST scheduling,
  // before any allocation: there is no mid-stream re-match/replacement on the
  // composite path. Lazily materializes the per-group state vector so the
  // policy always sees its cacheable groups. `tokens` is the full prompt; for
  // text-only sequences `mm_data` is null.
  virtual CompositeMatchResult match_prefix_cache(BlockManagerContext* context,
                                                  const Slice<int32_t>& tokens,
                                                  const MMData* mm_data =
                                                      nullptr);

  size_t num_groups() const { return runtimes_.size(); }

  // Schedulable capacity in base-block units, following the capacity rules of
  // the design doc: a C1 group contributes its own free blocks; otherwise
  // (DSV4) the binding constraint is min over the compressed incremental
  // groups of free_blocks * compress_ratio. SWA (windowed replacement) and
  // SINGLE_RES (per-sequence once) never join token-linear admission.
  virtual size_t num_free_blocks() const;

  // Free blocks of the allocator backing `state_id`; 0 when absent.
  virtual size_t group_free_blocks(CacheStateId state_id) const;

  // Blocks currently held out of the free list, summed across all group
  // allocators -- the composite's contribution to pool-level memory accounting.
  virtual size_t num_used_blocks() const;

  // Total blocks owned across all group allocators.
  virtual size_t num_total_blocks() const;

  // Blocks currently retained by the group-local prefix caches. The leaf
  // allocators run with prefix cache disabled, so this cached-block metric is
  // recovered here from each cacheable group's PrefixCache (the C1 entry for
  // normal models). Scheduler update_metrics paths read it.
  virtual size_t num_blocks_in_prefix_cache() const;

  // Fraction of owned blocks currently in use, in [0, 1]; 0 when empty.
  virtual double kv_cache_utilization() const;

 private:
  // Lazily size `kv_state`'s per-group state vector to match runtimes_,
  // stamping each entry's state_id. Idempotent across a sequence lifetime.
  void ensure_group_states(KVCacheState* kv_state) const;

  // Insert the sequence's newly-completed full blocks into the group-local
  // prefix caches, advancing each cacheable group's prefix_cached_tokens. This
  // is the internal "fill once per completed block" step: it runs at allocate
  // (lazily flushing the previous forward's committed blocks before growing)
  // and at deallocate (the final tail blocks before release). The committed-
  // token boundary is read from `context->kv_state->kv_cache_tokens_num()`.
  // No-op unless the context carries a hash_state and a non-empty token view
  // (see BlockManagerContext) and at least one group is prefix-cacheable.
  void insert_committed_blocks(BlockManagerContext* context);

  std::vector<CacheGroupRuntime> runtimes_;

  // Cross-group prefix match/insert. Exactly one per composite: IncrementalOnly
  // for normal (C1) models, NoPrefix when no group is prefix-cacheable.
  std::unique_ptr<ICompositePrefixPolicy> prefix_policy_;
};

}  // namespace xllm
