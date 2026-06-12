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
#include <memory>

#include "framework/block/block_manager_context.h"
#include "framework/block/cache_group.h"

namespace xllm {

class BlockManager;

// Per-cache-group allocation strategy. One instance serves one
// CacheGroupRuntime entry and is shared by all sequences of that manager, so
// calls must be serialized by the owning (Concurrent)CompositeBlockManager.
//
// rollback() undoes the most recent successful allocate() on the same state
// and must be invoked before any other policy call touches that state; the
// orchestrator uses it to revert already-allocated groups when a later group
// fails within one composite allocate.
class ICacheGroupPolicy {
 public:
  virtual ~ICacheGroupPolicy() = default;

  // Grow `state` to cover `num_tokens` committed tokens. Returns false and
  // leaves `state` unchanged when the allocator cannot satisfy the request.
  virtual bool allocate(BlockManagerContext* context,
                        CacheGroupState* state,
                        size_t num_tokens) = 0;

  // Release all blocks held by `state` and reset its bookkeeping.
  virtual void deallocate(BlockManagerContext* context,
                          CacheGroupState* state) = 0;

  virtual void rollback(BlockManagerContext* context,
                        CacheGroupState* state) = 0;
};

// Token-linear growth: needed_blocks = ceil(num_tokens / block_size), new
// blocks appended at the tail. Serves C1 / DSV4 C4 / DSV4 C128.
class IncrementalAppendPolicy final : public ICacheGroupPolicy {
 public:
  IncrementalAppendPolicy(const CacheGroupSpec& spec, BlockManager* allocator);

  bool allocate(BlockManagerContext* context,
                CacheGroupState* state,
                size_t num_tokens) override;
  void deallocate(BlockManagerContext* context,
                  CacheGroupState* state) override;
  void rollback(BlockManagerContext* context, CacheGroupState* state) override;

 private:
  CacheGroupSpec spec_;
  // non-owning, lifetime managed by the owning CacheGroupRuntime
  BlockManager* allocator_ = nullptr;
};

// Fixed-size ring for sliding-window states (DSV4 SWA): the first allocate
// occupies the full window_blocks ring in one shot; each later logical block
// crossing replaces slot = logical_block_idx % window_blocks in place. Slots
// are never compacted, shifted, or reordered — DSA metadata maps logical to
// physical positions by the same modulo and requires stable slot order.
//
// Replaced-out old blocks stay in state->pending_replacements until the next
// allocate on the same state commits them: if this allocate is rolled back,
// their KV is still inside the attention window of the old token end.
class RollingWindowPolicy final : public ICacheGroupPolicy {
 public:
  RollingWindowPolicy(const CacheGroupSpec& spec, BlockManager* allocator);

  bool allocate(BlockManagerContext* context,
                CacheGroupState* state,
                size_t num_tokens) override;
  void deallocate(BlockManagerContext* context,
                  CacheGroupState* state) override;
  void rollback(BlockManagerContext* context, CacheGroupState* state) override;

 private:
  CacheGroupSpec spec_;
  // non-owning, lifetime managed by the owning CacheGroupRuntime
  BlockManager* allocator_ = nullptr;
};

// One block per sequence, allocated on first use and reused afterwards.
// Serves SINGLE_RES (linear_state_ids / embedding_ids / MTP); never
// participates in prefix cache.
class PerSequenceOncePolicy final : public ICacheGroupPolicy {
 public:
  PerSequenceOncePolicy(const CacheGroupSpec& spec, BlockManager* allocator);

  bool allocate(BlockManagerContext* context,
                CacheGroupState* state,
                size_t num_tokens) override;
  void deallocate(BlockManagerContext* context,
                  CacheGroupState* state) override;
  void rollback(BlockManagerContext* context, CacheGroupState* state) override;

 private:
  CacheGroupSpec spec_;
  // non-owning, lifetime managed by the owning CacheGroupRuntime
  BlockManager* allocator_ = nullptr;
};

// Factory keyed by spec.policy_type.
std::unique_ptr<ICacheGroupPolicy> create_cache_group_policy(
    const CacheGroupSpec& spec,
    BlockManager* allocator);

}  // namespace xllm
