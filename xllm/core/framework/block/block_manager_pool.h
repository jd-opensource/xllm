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

#include <queue>
#include <vector>

#include "block_manager.h"
#include "framework/block/kv_cache_manager.h"

namespace xllm {

class BlockManagerPool : public KVCacheManager {
 public:
  struct Options {
    PROPERTY(uint32_t, num_blocks) = 0;
    PROPERTY(uint32_t, host_num_blocks) = 0;
    PROPERTY(int32_t, block_size) = 0;
    PROPERTY(bool, enable_prefix_cache) = true;
    PROPERTY(bool, enable_disagg_pd) = false;
    PROPERTY(bool, enable_cache_upload) = false;
    PROPERTY(bool, enable_kvcache_store) = false;
  };

  explicit BlockManagerPool(const Options& options, int32_t dp_size = 1);

  ~BlockManagerPool() = default;

  virtual bool allocate(Sequence* sequence) override;
  virtual bool allocate(std::vector<Sequence*>& sequences) override;
  virtual bool allocate(Sequence* sequence, size_t num_tokens) override;

  // Try to allocate blocks with num_tokens,
  // return {} if not enough blocks
  virtual std::vector<Block> allocate(size_t num_tokens,
                                      int32_t& dp_rank) override;

  virtual bool try_allocate(Sequence* sequence) override;

  virtual void deallocate(Request* request) override;
  virtual void deallocate(std::vector<Sequence*>& sequences) override;
  virtual void deallocate(Sequence* sequence) override;

  virtual void allocate_shared(Sequence* sequence) override;
  virtual void cache(Sequence* sequence) override;

  virtual std::vector<std::vector<BlockTransferInfo>>*
  get_swap_block_transfer_infos() override;
  virtual void reset_transfer_infos() override;

  virtual void get_merged_kvcache_event(KvCacheEvent* event) const;
  virtual float get_gpu_cache_usage_perc() const;

  virtual uint32_t num_blocks() const override;
  virtual int32_t block_size() const override;
  virtual std::vector<size_t> num_blocks_in_prefix_cache() const override;
  virtual std::vector<size_t> num_free_blocks() const override;
  virtual std::vector<size_t> num_used_blocks() const override;
  virtual double kv_cache_utilization() const override;

  // get the options for the block manager
  const Options& options() const { return options_; }

 protected:
  int32_t get_manager_with_max_free_blocks() const;
  int32_t get_dp_rank(Sequence* sequence) const;

  bool process_beam_search(Sequence* sequence, bool need_swap = false);

 private:
  std::vector<std::vector<BlockTransferInfo>> swap_block_transfer_infos_;

 protected:
  // the options for the block manager
  Options options_;
  std::vector<std::unique_ptr<BlockManager>> block_managers_;
};

}  // namespace xllm
