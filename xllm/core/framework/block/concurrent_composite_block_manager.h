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
#include <mutex>
#include <vector>

#include "framework/block/block_manager_context.h"
#include "framework/block/cache_group.h"
#include "framework/block/group_composite_block_manager.h"

namespace xllm {

// Concurrency wrapper around GroupCompositeBlockManager. Modes whose
// sequence-level entry points run off the scheduler thread -- disagg PD and
// pd_ooc prefill threadpools, kvcache store, the host pool -- call allocate /
// deallocate concurrently with each other; this wrapper serializes them under
// one lock. It takes over the lock duty that ConcurrentBlockManagerImpl used
// to hold for the monolithic path.
//
// A plain (non-recursive) mutex suffices: the design forbids the leaf free
// path from re-entering the composite layer (Block destructors call the leaf
// allocator's own lock directly), and composite -> leaf is the only ordering,
// so the lock is never re-acquired by a nested call. Once per-group prefix
// caches land, their eviction releases Blocks while this lock is held, but
// those releases still terminate at the leaf allocator, never back here.
class ConcurrentCompositeBlockManager final {
 public:
  explicit ConcurrentCompositeBlockManager(
      const std::vector<CacheGroupSpec>& specs);

  bool allocate(BlockManagerContext* context, size_t num_tokens);

  void deallocate(BlockManagerContext* context);

  // Match the prompt prefix against the group-local prefix caches. Serialized
  // under the same lock as allocate/deallocate because the match attaches
  // shared Blocks to the leaf allocators' bookkeeping.
  CompositeMatchResult match_prefix_cache(BlockManagerContext* context,
                                          const Slice<int32_t>& tokens,
                                          const MMData* mm_data = nullptr);

  size_t num_groups() const { return composite_.num_groups(); }

  size_t num_free_blocks() const;

  size_t group_free_blocks(CacheStateId state_id) const;

  size_t num_used_blocks() const;

  size_t num_total_blocks() const;

  size_t num_blocks_in_prefix_cache() const;

  double kv_cache_utilization() const;

 private:
  GroupCompositeBlockManager composite_;

  // Serializes sequence-level allocate/deallocate across worker threads.
  // Composite -> leaf allocator is the only lock ordering; the leaf lock never
  // calls back up, so this stays a plain mutex.
  mutable std::mutex mutex_;
};

}  // namespace xllm
