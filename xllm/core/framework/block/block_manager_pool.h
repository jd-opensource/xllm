/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <vector>

#include "block_manager.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"

namespace xllm {

class BlockManagerPool {
 public:
  explicit BlockManagerPool(const BlockManager::Options& options,
                            int32_t dp_size = 1);

  ~BlockManagerPool() = default;

  bool allocate(Sequence* sequence);
  bool allocate(std::vector<Sequence*>& sequences);
  bool allocate(Sequence* sequence, size_t num_tokens);

  // Try to allocate blocks with num_tokens,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_tokens, int32_t& dp_rank);

  void deallocate(Request* request);
  void deallocate(std::vector<Sequence*>& sequences);
  void deallocate(Sequence* sequence);

  void allocate_shared(Sequence* sequence);
  void cache(Sequence* sequence);

  void get_merged_kvcache_event(KvCacheEvent* event) const;
  float get_gpu_cache_usage_perc() const;

  bool check_if_enough_to_evict(DecodePriorityQueue* running_queue_to_evict,
                                Sequence* prefill_sequence,
                                size_t& num_request_to_evict);

  std::vector<size_t> num_blocks_in_prefix_cache() const;
  std::vector<size_t> num_free_blocks() const;
  std::vector<size_t> num_used_blocks() const;
  double kv_cache_utilization() const;

  // get the options for the block manager
  const BlockManager::Options& options() const { return options_; }

 private:
  int32_t get_manager_with_max_free_blocks() const;
  int32_t get_dp_rank(Sequence* sequence) const;

  std::vector<std::unique_ptr<BlockManager>> block_managers_;

  // the options for the block manager
  BlockManager::Options options_;
};

}  // namespace xllm
