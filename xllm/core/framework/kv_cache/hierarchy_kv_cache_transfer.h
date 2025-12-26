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

#include <torch/torch.h>

#include <memory>

#include "common/types.h"
#include "framework/model/model_input_params.h"
#include "kv_cache.h"
#include "platform/device.h"
#include "util/blockingconcurrentqueue.h"
#include "util/threadpool.h"

#if defined(USE_NPU)
#include "acl/acl_rt.h"
#include "platform/npu/npu_layer_synchronizer.h"
#endif

namespace xllm {
class HierarchyKVCacheTransfer {
 public:
  struct Options {
    PROPERTY(uint32_t, tp_rank);
    PROPERTY(uint32_t, layers);
    PROPERTY(double, host_blocks_factor) = 0.0;
    PROPERTY(uint32_t, layers_wise_copy_batchs) = 1;
    PROPERTY(bool, enable_kvcache_store) = false;
    PROPERTY(std::string, store_protocol) = "tcp";
    PROPERTY(std::string, store_master_server_address) = "";
    PROPERTY(std::string, store_metadata_server) = "";
    PROPERTY(std::string, store_local_hostname) = "";
  };

  HierarchyKVCacheTransfer(const Options& options,
                           const torch::Device& device,
                           std::vector<xllm::KVCache>* kv_caches_ptr);
  ~HierarchyKVCacheTransfer();

  uint32_t transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  uint32_t transfer_kv_blocks(const uint64_t batch_id,
                              Slice<BlockTransferInfo>& block_transfer_info);

  void set_layer_synchronizer(ModelInputParams& params);

 private:
  void create_page_aligned_host_cache();

  uint32_t offload_kv_blocks(
      const std::vector<BlockTransferInfo>& block_transfer_info);

  bool d2h_batch_copy(Slice<BlockTransferInfo>& block_transfer_info);
  bool h2d_batch_copy(const uint64_t batch_id,
                      Slice<BlockTransferInfo>& block_transfer_info);

  uint32_t offload_to_store(Slice<BlockTransferInfo>& block_transfer_info);
  uint32_t load_from_store(Slice<BlockTransferInfo>& block_transfer_info);

 private:
  // options
  Options options_;
  // device to run the model on
  Device device_;

  // working thread for data copy
  std::unique_ptr<ThreadPool> h2d_threadpool_;
  std::unique_ptr<ThreadPool> d2h_threadpool_;
  // copy streams only can be used in h2d_threadpool_ and d2h_threadpool_
  moodycamel::BlockingConcurrentQueue<std::unique_ptr<Stream>> copy_stream_;

  std::vector<xllm::KVCache>* kv_caches_ptr_;
  std::vector<xllm::KVCache> host_kv_caches_;

  void* page_aligned_data_ = nullptr;
  size_t page_aligned_data_size_ = 0;

  std::vector<uint64_t> cache_size_per_layer_;
  uint32_t cache_tensor_cnt_ = 1;

#if defined(USE_NPU)
  void* mapped_ptr_ = nullptr;

  aclrtMemcpyBatchAttr h2d_attrs_;
  aclrtMemcpyBatchAttr d2h_attrs_;

  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<NPULayerSynchronizerImpl>>
      layer_wise_load_synchronizer_;
#endif
};

}  // namespace xllm
