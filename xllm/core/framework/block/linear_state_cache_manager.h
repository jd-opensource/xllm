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

#include <cstdint>
#include <memory>
#include <vector>

#include "common/macros.h"
#include "framework/prefix_cache/linear_state_prefix_cache.h"

namespace xllm {

class Sequence;
struct LinearStateCacheOp;

class LinearStateCacheManager final {
 public:
  struct Options {
    PROPERTY(int32_t, num_slots) = 0;
    PROPERTY(int32_t, dp_size) = 1;
    PROPERTY(int32_t, block_size) = 0;
  };

  explicit LinearStateCacheManager(const Options& options);

  bool allocate_slot(Sequence* sequence, int32_t dp_rank);
  void release_slot(Sequence* sequence);

  LinearStatePrefixCache* prefix_cache(int32_t dp_rank);

  LinearStateCheckpointReservations resolve_cache_ops(
      int32_t dp_rank,
      std::vector<LinearStateCacheOp>* cache_ops,
      const std::vector<Sequence*>& sequences = {});

  void commit_reservations(LinearStateCheckpointReservations&& reservations);

  // Compute the number of shared KV blocks that have a corresponding
  // linear-state checkpoint. The caller is responsible for deallocating
  // blocks beyond this count.
  size_t compute_safe_shared_prefix_length(int32_t dp_rank,
                                           const Sequence* sequence,
                                           size_t existed_shared_blocks_num,
                                           size_t total_shared_blocks) const;

 private:
  Options options_;
  std::vector<std::unique_ptr<LinearStatePrefixCache>> caches_;
};

}  // namespace xllm
