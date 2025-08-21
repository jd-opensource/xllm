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

#include "framework/block/block_manager_pool.h"
#include "framework/page/page_manager_pool.h"

namespace xllm {

class KVCacheManagerClient {
 public:
  KVCacheManagerClient(BlockManagerPool* block_manager_pool);
  KVCacheManagerClient(PageManagerPool* page_manager_pool);

  ~KVCacheManagerClient() = default;

  bool allocate(Sequence* sequence);
  bool allocate(std::vector<Sequence*>& sequences);
  bool allocate(Sequence* sequence, size_t num_tokens);

  uint32_t pre_allocate(Sequence* sequence);

  std::vector<Block> allocate(size_t num_tokens, int32_t& dp_rank);

  void deallocate(Request* request);
  void deallocate(std::vector<Sequence*>& sequences);
  void deallocate(Sequence* sequence);

  void allocate_shared(Sequence* sequence);
  void cache(Sequence* sequence);

  std::vector<std::vector<CacheBlockInfo>>* get_copy_in_cache_block_infos();
  std::vector<std::vector<CacheBlockInfo>>* get_copy_out_cache_block_infos();
  void reset_copy_content();

  std::vector<size_t> num_blocks_in_prefix_cache() const;
  std::vector<size_t> num_free_blocks() const;
  std::vector<size_t> num_used_blocks() const;
  double kv_cache_utilization() const;

  const BlockManagerPool::Options& options() const;

 private:
  // these two pointers must be one null and one non-null
  BlockManagerPool* block_manager_pool_;
  PageManagerPool* page_manager_pool_;
};
}  // namespace xllm