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

#include "block_manager_impl.h"

namespace xllm {

// Allocates exactly options_.window_size() blocks per sequence.
// Used as a sub-manager of CompositeBlockManager.
class SlidingWindowBlockManager : public BlockManagerImpl {
 public:
  explicit SlidingWindowBlockManager(const Options& options);
  ~SlidingWindowBlockManager() override = default;

  // Only allocation of window_size() blocks is allowed.
  std::vector<Block> allocate(size_t num_blocks) override;

  uint32_t window_size() const { return options_.window_size(); }
};

}  // namespace xllm
