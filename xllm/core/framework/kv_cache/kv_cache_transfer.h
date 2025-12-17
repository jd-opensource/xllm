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

#include "common/types.h"
#include "kv_cache.h"
#if defined(USE_NPU)
#include "platform/npu/npu_layer_synchronizer.h"
#endif
#include "framework/parallel_state/parallel_args.h"
#include "util/threadpool.h"

namespace xllm {

class KVCacheTransfer {
 public:
  struct KVCacheInfo {
    uint64_t dst_cluster_id;
    std::string dst_addr;
    int64_t dst_k_cache_id;
    int64_t dst_v_cache_id;
    std::vector<uint64_t> src_blocks;
    std::vector<uint64_t> dst_blocks;
    std::vector<uint64_t> src_embed_ids;
    std::vector<uint64_t> dst_embed_ids;
  };

  KVCacheTransfer() = default;
  virtual ~KVCacheTransfer() = default;

  virtual void initialize(int32_t device_id) {};

  virtual void finalize() {};

  virtual void allocate_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      const torch::ScalarType dtype) {};

  virtual void allocate_kv_cache_spec(
      std::vector<xllm::KVCache>& kv_caches,
      const int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      torch::ScalarType dtype) {
    LOG(FATAL) << "allocate_kv_cache_spec not implemented for KVCacheTransfer!";
  };

  virtual void free_kv_cache() {};

  virtual void register_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      const torch::ScalarType dtype) {};

  virtual void get_cache_info(uint64_t& cluster_id,
                              std::string& addr,
                              int64_t& key_cache_id,
                              int64_t& value_cache_id) = 0;

  virtual bool link_cluster(const uint64_t cluster_id,
                            const std::string& remote_addr,
                            const std::string& device_ip,
                            const uint16_t port) = 0;

  virtual bool unlink_cluster(const uint64_t& cluster_id,
                              const std::string& remote_addr,
                              const std::string& device_ip,
                              const uint16_t port,
                              bool force_flag = false) = 0;

  virtual bool pull_kv_blocks(const uint64_t src_cluster_id,
                              const std::string& src_addr,
                              const int64_t src_k_cache_id,
                              const int64_t src_v_cache_id,
                              const std::vector<uint64_t>& src_blocks,
                              const std::vector<uint64_t>& dst_blocks) = 0;

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const int64_t src_k_cache_id,
      const int64_t src_v_cache_id,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks);

#if defined(USE_NPU)
  virtual folly::SemiFuture<bool> push_kv_blocks_async(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args,
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer,
      bool is_spec_draft);
#endif

  virtual void merge_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args);

#if defined(USE_NPU)
  virtual bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft) = 0;
#endif

  virtual std::vector<torch::Tensor> convert_to_torch_tensor(
      const std::vector<int64_t>& dims,
      const torch::ScalarType dtype,
      const std::vector<uintptr_t>& addresses);

 protected:
  // working thread
  ThreadPool threadpool_;
};

}  // namespace xllm
