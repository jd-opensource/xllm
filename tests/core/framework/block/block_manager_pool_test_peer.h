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

#pragma once

#include "framework/block/block_manager_pool.h"
#include "framework/block/composite_block_manager.h"
#include "framework/block/linear_state_block_manager.h"

namespace xllm {

// Test-only accessor for BlockManagerPool internals. Exposes the private LINEAR
// leaf so tests can seed checkpoints directly the way the scheduler does while
// resolving cache ops, without widening the production API surface.
class BlockManagerPoolTestPeer final {
 public:
  static LinearStateBlockManager* linear_leaf(BlockManagerPool& pool,
                                              int32_t dp_rank) {
    if (!pool.options_.enable_linear_state()) {
      return nullptr;
    }
    auto* composite = static_cast<CompositeBlockManager*>(
        pool.block_managers_[dp_rank].get());
    return static_cast<LinearStateBlockManager*>(
        composite->leaf_of(BlockType::LINEAR));
  }
};

}  // namespace xllm
