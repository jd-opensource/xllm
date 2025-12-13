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

#include "common/macros.h"
#include "framework/batch/batch.h"
#include "framework/model/model_input_params.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"

namespace xllm {

class KVCacheManager {
 public:
  virtual ~KVCacheManager() = default;

  virtual bool allocate(Sequence* sequence) = 0;
  virtual bool allocate(std::vector<Sequence*>& sequences) = 0;
  virtual bool allocate(Sequence* sequence, size_t num_tokens) = 0;

  virtual void transfer_blocks(std::optional<std::vector<Batch>> batches) {
    return;
  };

  virtual void prefetch_from_storage(std::shared_ptr<Request>& request) {
    return;
  };

  virtual bool update_prefetch_result(std::shared_ptr<Request>& request,
                                      const uint32_t timeout) {
    return true;
  };

  virtual std::vector<Block> allocate(size_t num_tokens, int32_t& dp_rank) = 0;

  virtual void deallocate(Request* request) = 0;
  virtual void deallocate(std::vector<Sequence*>& sequences) = 0;
  virtual void deallocate(Sequence* sequence) = 0;

  virtual void allocate_shared(Sequence* sequence) = 0;
  virtual void cache(Sequence* sequence) = 0;

  virtual std::vector<std::vector<BlockTransferInfo>>*
  get_swap_block_transfer_infos() = 0;

  virtual void reset_transfer_infos() = 0;

  virtual uint32_t num_blocks() const = 0;
  virtual int32_t block_size() const = 0;
  virtual std::vector<size_t> num_blocks_in_prefix_cache() const = 0;
  virtual std::vector<size_t> num_free_blocks() const = 0;
  virtual std::vector<size_t> num_used_blocks() const = 0;
  virtual double kv_cache_utilization() const = 0;

 protected:
  KVCacheManager() = default;

 private:
  DISALLOW_COPY_AND_ASSIGN(KVCacheManager);
};

}  // namespace xllm
