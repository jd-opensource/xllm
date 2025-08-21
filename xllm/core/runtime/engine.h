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

#include <folly/futures/Future.h>

#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/page/page_manager_pool.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "options.h"

namespace xllm {
class Engine {
 public:
  virtual ~Engine() = default;

  virtual bool init() = 0;

  // execute model with batch input
  virtual ForwardOutput step(std::vector<Batch>& batch) = 0;

  virtual void update_last_step_result(std::vector<Batch>& batch) = 0;

  // return the tokenizer
  virtual const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  // return the block manager
  virtual BlockManagerPool* block_manager_pool() const {
    return block_manager_pool_.get();
  }

  virtual PageManagerPool* page_manager_pool() const {
    return page_manager_pool_.get();
  }

  // return the model args
  virtual const ModelArgs& model_args() const { return args_; }

  // return the tokenizer args
  virtual const TokenizerArgs& tokenizer_args() const {
    return tokenizer_args_;
  }

  // return the active activation memory
  virtual std::vector<int64_t> get_active_activation_memory() const = 0;

  // P/D
  virtual bool pull_kv_blocks(const int32_t src_dp_size,
                              const int32_t src_dp_rank,
                              const std::vector<uint64_t>& src_cluster_ids,
                              const std::vector<std::string>& src_addrs,
                              const std::vector<int64_t>& src_k_cache_ids,
                              const std::vector<int64_t>& src_v_cache_ids,
                              const std::vector<uint64_t>& src_blocks,
                              const int32_t dst_dp_rank,
                              const std::vector<uint64_t>& dst_blocks) {
    LOG(FATAL) << " pull_kv_blocks is notimplemented!";
  };

  virtual std::vector<folly::SemiFuture<uint32_t>>
  load_kv_blocks_from_store_async(
      const uint32_t dp_rank,
      const std::vector<CacheBlockInfo>& cache_block_info) {
    LOG(FATAL) << " load_kv_blocks_from_store is not implemented!";
  };

  virtual void get_device_info(std::vector<std::string>& device_ips,
                               std::vector<uint16_t>& ports) {
    LOG(FATAL) << " get_device_info is notimplemented!";
  };

  virtual void get_cache_info(std::vector<uint64_t>& cluster_ids,
                              std::vector<std::string>& addrs,
                              std::vector<int64_t>& k_cache_ids,
                              std::vector<int64_t>& v_cache_ids) {
    LOG(FATAL) << " get_cache_info is notimplemented!";
  };

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<std::string>& device_ips,
                            const std::vector<uint16_t>& ports,
                            const int32_t src_dp_size) {
    LOG(FATAL) << " link_cluster is notimplemented!";
  };

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<std::string>& device_ips,
                              const std::vector<uint16_t>& ports,
                              const int32_t dp_size) {
    LOG(FATAL) << " unlink_cluster is notimplemented!";
  };

  struct KVCacheCapacity {
    int64_t n_blocks = 0;
    int64_t n_pages = 0;  // for continuous kvcache
    int64_t cache_size_in_bytes = 0;
    int64_t slot_size = 0;
  };

 protected:
  // model args
  ModelArgs args_;

  // Tokenizer args
  TokenizerArgs tokenizer_args_;

  // block manager
  std::unique_ptr<BlockManagerPool> block_manager_pool_;

  // page manager
  std::unique_ptr<PageManagerPool> page_manager_pool_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
};
}  // namespace xllm
