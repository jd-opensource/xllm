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

#include "group_composite_block_manager.h"

#include <glog/logging.h>

#include <utility>

#include "framework/block/block_manager_impl.h"
#include "framework/prefix_cache/prefix_cache.h"
#include "framework/prefix_cache/prefix_hash_state.h"
#include "framework/request/sequence_kv_state.h"

namespace xllm {
namespace {

std::unique_ptr<BlockManager> create_group_allocator(
    const CacheGroupSpec& spec) {
  BlockManager::Options options;
  options.num_blocks(spec.num_blocks)
      .block_size(static_cast<int32_t>(spec.block_size))
      // The composite owns cross-group matching/caching through its prefix
      // policy and a separate group-local PrefixCache; the leaf allocator runs
      // as a pure block pool, accounting shared blocks purely via ref counts.
      .enable_prefix_cache(false);
  return std::make_unique<BlockManagerImpl>(options);
}

}  // namespace

GroupCompositeBlockManager::GroupCompositeBlockManager(
    const std::vector<CacheGroupSpec>& specs) {
  CHECK(!specs.empty()) << "composite manager needs at least one cache group";
  runtimes_.reserve(specs.size());
  for (const CacheGroupSpec& spec : specs) {
    CacheGroupRuntime runtime;
    runtime.spec = spec;
    runtime.allocator = create_group_allocator(spec);
    runtime.policy = create_cache_group_policy(spec, runtime.allocator.get());
    if (spec.prefix_cacheable) {
      CHECK(spec.prefix_group != PrefixCacheGroup::INVALID)
          << "prefix-cacheable group " << to_string(spec.state_id)
          << " must declare a prefix_group";
      // Hash stride equals the group's block_size by construction.
      runtime.prefix_cache = std::make_unique<PrefixCache>(spec.block_size);
    }
    runtimes_.emplace_back(std::move(runtime));
  }

  // Build the composite prefix policy over the (now address-stable) runtimes.
  std::vector<CacheablePrefixEntry> cacheable;
  for (const CacheGroupRuntime& runtime : runtimes_) {
    if (runtime.prefix_cache == nullptr) {
      continue;
    }
    CacheablePrefixEntry entry;
    entry.state_id = runtime.spec.state_id;
    entry.group = runtime.spec.prefix_group;
    entry.block_size = runtime.spec.block_size;
    entry.cache = runtime.prefix_cache.get();
    cacheable.push_back(entry);
  }

  if (cacheable.empty()) {
    prefix_policy_ = std::make_unique<NoPrefixPolicy>();
  } else {
    // Phase 1 supports the normal-model shape only: a single C1 cache.
    // Multi-group (DSV4 C4/C128) policies arrive in a later migration step.
    CHECK_EQ(cacheable.size(), 1u)
        << "only a single prefix-cacheable group is supported in phase 1";
    CHECK(cacheable[0].group == PrefixCacheGroup::C1)
        << "phase-1 prefix caching expects the C1 group, got "
        << cacheable[0].group.to_string();
    prefix_policy_ =
        std::make_unique<IncrementalOnlyPrefixPolicy>(cacheable[0]);
  }
}

GroupCompositeBlockManager::~GroupCompositeBlockManager() = default;

void GroupCompositeBlockManager::ensure_group_states(
    KVCacheState* kv_state) const {
  CHECK(kv_state != nullptr);
  std::vector<CacheGroupState>* groups = kv_state->mutable_groups();
  if (groups->size() == runtimes_.size()) {
    return;
  }
  CHECK(groups->empty()) << "group state count " << groups->size()
                         << " does not match composite group count "
                         << runtimes_.size();
  groups->reserve(runtimes_.size());
  for (const CacheGroupRuntime& runtime : runtimes_) {
    CacheGroupState state;
    state.state_id = runtime.spec.state_id;
    groups->emplace_back(std::move(state));
  }
}

bool GroupCompositeBlockManager::allocate(BlockManagerContext* context,
                                          size_t num_tokens) {
  CHECK(context != nullptr);
  CHECK(context->kv_state != nullptr);

  KVCacheState* kv_state = context->kv_state;
  ensure_group_states(kv_state);

  // Lazy flush: the blocks the previous forward pass completed are now
  // committed, so insert them into the prefix caches before growing further.
  // No-op on a fresh sequence (nothing committed) and on bare contexts.
  insert_committed_blocks(context);

  std::vector<CacheGroupState>* groups = kv_state->mutable_groups();
  CHECK_EQ(groups->size(), runtimes_.size());

  for (size_t i = 0; i < runtimes_.size(); ++i) {
    CacheGroupRuntime& runtime = runtimes_[i];
    bool ok = runtime.policy->allocate(context, &(*groups)[i], num_tokens);
    if (!ok && runtime.prefix_cache != nullptr) {
      // Group-local allocation fallback: the leaf pool ran dry, but this group
      // owns a prefix cache pinning completed blocks out of the free list.
      // Evict exactly the shortfall from THIS group's cache (never another
      // group's) and retry once. Evicted blocks are not restored if a later
      // group in the same allocate fails -- the prefix cache is a best-effort
      // reuse pool, not part of the all-or-nothing block reservation.
      const size_t needed =
          runtime.policy->additional_blocks_needed((*groups)[i], num_tokens);
      const size_t free = runtime.allocator->num_free_blocks();
      if (needed > free) {
        runtime.prefix_cache->evict(needed - free);
      }
      ok = runtime.policy->allocate(context, &(*groups)[i], num_tokens);
    }
    if (ok) {
      continue;
    }
    // Atomic across groups: undo the groups already grown in this call. The
    // failing group already rolled its own partial work back internally.
    for (size_t j = i; j-- > 0;) {
      runtimes_[j].policy->rollback(context, &(*groups)[j]);
    }
    return false;
  }
  return true;
}

void GroupCompositeBlockManager::deallocate(BlockManagerContext* context) {
  CHECK(context != nullptr);
  CHECK(context->kv_state != nullptr);

  std::vector<CacheGroupState>* groups = context->kv_state->mutable_groups();
  if (groups->empty()) {
    return;
  }
  CHECK_EQ(groups->size(), runtimes_.size());

  // Final tail flush: insert the last completed blocks before releasing. Bare
  // contexts (preempt) skip this and release without caching.
  insert_committed_blocks(context);

  for (size_t i = 0; i < runtimes_.size(); ++i) {
    runtimes_[i].policy->deallocate(context, &(*groups)[i]);
  }
  groups->clear();
}

CompositeMatchResult GroupCompositeBlockManager::match_prefix_cache(
    BlockManagerContext* context,
    const Slice<int32_t>& tokens,
    const MMData* mm_data) {
  CHECK(context != nullptr);
  CHECK(context->kv_state != nullptr);
  // The policy reads each cacheable group's per-sequence state, so it must
  // exist (empty is fine) before matching.
  ensure_group_states(context->kv_state);

  PrefixCacheMatchContext match_ctx;
  match_ctx.kv_state = context->kv_state;
  match_ctx.tokens = tokens;
  match_ctx.mm_data = mm_data;
  match_ctx.role = context->role;
  match_ctx.device_dp_rank = context->device_dp_rank;
  return prefix_policy_->match(match_ctx);
}

void GroupCompositeBlockManager::insert_committed_blocks(
    BlockManagerContext* context) {
  CHECK(context != nullptr);
  CHECK(context->kv_state != nullptr);

  // The internal insert only runs when the caller wired the token view and the
  // prefix-hash chain. The preempt path (deallocate_without_cache) deliberately
  // passes neither, so its release never writes uncomputed blocks to the cache.
  if (context->hash_state == nullptr || context->tokens.empty()) {
    return;
  }

  PrefixCacheInsertContext insert_ctx;
  insert_ctx.kv_state = context->kv_state;
  insert_ctx.tokens = context->tokens;
  // The committed boundary is the KV the forward pass has actually written.
  insert_ctx.committed_tokens = context->kv_state->kv_cache_tokens_num();
  insert_ctx.hash_state = context->hash_state;
  insert_ctx.role = context->role;
  insert_ctx.device_dp_rank = context->device_dp_rank;
  prefix_policy_->insert_committed(insert_ctx);
}

size_t GroupCompositeBlockManager::num_free_blocks() const {
  // Reads the leaf allocators' atomic counters; no composite lock needed.
  size_t total = 0;
  for (const CacheGroupRuntime& runtime : runtimes_) {
    total += runtime.allocator->num_free_blocks();
  }
  return total;
}

size_t GroupCompositeBlockManager::group_free_blocks(
    CacheStateId state_id) const {
  for (const CacheGroupRuntime& runtime : runtimes_) {
    if (runtime.spec.state_id == state_id) {
      return runtime.allocator->num_free_blocks();
    }
  }
  return 0;
}

size_t GroupCompositeBlockManager::num_used_blocks() const {
  size_t total = 0;
  for (const CacheGroupRuntime& runtime : runtimes_) {
    total += runtime.allocator->num_used_blocks();
  }
  return total;
}

size_t GroupCompositeBlockManager::num_total_blocks() const {
  size_t total = 0;
  for (const CacheGroupRuntime& runtime : runtimes_) {
    total += runtime.allocator->num_total_blocks();
  }
  return total;
}

size_t GroupCompositeBlockManager::num_blocks_in_prefix_cache() const {
  // Leaf allocators are pure pools (enable_prefix_cache=false), so the
  // cached-block count lives in each cacheable group's own PrefixCache.
  size_t total = 0;
  for (const CacheGroupRuntime& runtime : runtimes_) {
    if (runtime.prefix_cache != nullptr) {
      total += runtime.prefix_cache->num_blocks();
    }
  }
  return total;
}

double GroupCompositeBlockManager::kv_cache_utilization() const {
  const size_t total = num_total_blocks();
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(num_used_blocks()) / static_cast<double>(total);
}

}  // namespace xllm
