/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "block.h"
#include "common/global_flags.h"
#include "common/macros.h"
#include "common/metrics.h"
#include "common/types.h"
#include "framework/prefix_cache/prefix_cache.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "scheduler/decode_priority_queue.h"
#include "util/timer.h"

namespace xllm {
// class DecodePriorityQueue;
class BlockManager {
 public:
  struct Options {
    PROPERTY(uint32_t, num_blocks) = 0;
    PROPERTY(int32_t, block_size) = 0;
    PROPERTY(bool, enable_prefix_cache) = true;
    PROPERTY(bool, enable_disagg_pd) = false;
    PROPERTY(bool, enable_cache_upload) = false;
  };

  explicit BlockManager(Options options) : options_(options) {}
  virtual ~BlockManager() = default;

  virtual void deallocate(const Slice<Block>& blocks) = 0;

  virtual std::vector<Block> allocate(size_t num_blocks) = 0;

  virtual std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {}) = 0;

  virtual void cache(const Slice<int32_t>& token_ids,
                     std::vector<Block>& blocks) = 0;
  virtual void cache(const std::vector<Block>& blocks) = 0;

  // get merged all dp rank KVCacheEvent
  virtual void get_merged_kvcache_event(KvCacheEvent* event) const = 0;

  virtual size_t num_blocks_in_prefix_cache() const = 0;
  virtual size_t num_free_blocks() const = 0;
  virtual size_t num_used_blocks() const = 0;
  virtual double kv_cache_utilization() const = 0;

  // get the options for the block manager
  const Options& options() const { return options_; }

  // get number of slots per block
  size_t block_size() const { return options_.block_size(); }

  // call BlockManager to free block used by Block.
  virtual void free(int32_t block_id) = 0;

  // allocate a list of blocks, used for unit test
  // virtual std::vector<Block> allocate(uint32_t n_blocks) = 0;

  // allocate a block, used for unit test
  virtual Block allocate() = 0;

  // get number of total blocks
  virtual size_t num_total_blocks() const = 0;

 protected:
  // the options for the block manager
  Options options_;
};

}  // namespace xllm
