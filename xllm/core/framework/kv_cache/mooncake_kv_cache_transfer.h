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

#include "kv_cache_transfer.h"
#include "mooncake_transfer_engine.h"

namespace xllm {

class MooncakeKVCacheTransfer : public KVCacheTransfer {
 public:
  MooncakeKVCacheTransfer(const int32_t device_id,
                          const int16_t listen_port,
                          const torch::Device& device);
  virtual ~MooncakeKVCacheTransfer() = default;

  virtual void initialize(int32_t device_id) override;

  virtual void allocate_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      torch::ScalarType dtype) override;

  virtual void register_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      const torch::ScalarType dtype) override;

  virtual void get_cache_info(uint64_t& cluster_id,
                              std::string& addr,
                              int64_t& key_cache_id,
                              int64_t& value_cache_id) override;

  virtual bool link_cluster(const uint64_t cluster_id,
                            const std::string& remote_addr,
                            const std::string& device_ip,
                            const uint16_t port) override;

  virtual bool unlink_cluster(const uint64_t& cluster_id,
                              const std::string& remote_addr,
                              const std::string& device_ip,
                              const uint16_t port,
                              bool force_flag = false) override;

  virtual bool pull_kv_blocks(const uint64_t src_cluster_id,
                              const std::string& src_addr,
                              const int64_t src_k_cache_id,
                              const int64_t src_v_cache_id,
                              const std::vector<uint64_t>& src_blocks,
                              const std::vector<uint64_t>& dst_blocks) override;

  virtual bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft) override;

 private:
  std::string addr_;
  uint64_t cluster_id_;
  int16_t listen_port_;
  int32_t device_id_;
  int64_t num_layers_;

  std::unique_ptr<MooncakeTransferEngine> mooncake_te_;
};

}  // namespace xllm
