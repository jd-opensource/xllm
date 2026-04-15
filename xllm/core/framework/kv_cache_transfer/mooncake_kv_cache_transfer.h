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

#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

namespace xllm {

// Base class for Mooncake-based KV cache transfer.
// Default and XTensor subclasses inherit this class (single inheritance).
class MooncakeKVCacheTransferBase : public KVCacheTransfer {
 public:
  MooncakeKVCacheTransferBase(const int32_t device_id,
                              const int16_t listen_port,
                              const torch::Device& device,
                              std::unique_ptr<MooncakeTransferEngine> engine);
  ~MooncakeKVCacheTransferBase() override = default;

  void initialize(int32_t device_id) override;

  void get_cache_info(uint64_t& cluster_id,
                      std::string& addr,
                      int64_t& key_cache_id,
                      int64_t& value_cache_id) override;

  bool link_cluster(const uint64_t cluster_id,
                    const std::string& remote_addr,
                    const std::string& device_ip,
                    const uint16_t port) override;

  bool unlink_cluster(const uint64_t& cluster_id,
                      const std::string& remote_addr,
                      const std::string& device_ip,
                      const uint16_t port,
                      bool force_flag = false) override;

 protected:
  std::string addr_;
  uint64_t cluster_id_;
  int16_t listen_port_;
  int32_t device_id_;
  torch::Device device_;
  int64_t num_layers_ = 0;
  int64_t size_per_block_ = 0;

  std::unique_ptr<MooncakeTransferEngine> mooncake_te_;
};

class MooncakeKVCacheTransferDefault final
    : public MooncakeKVCacheTransferBase {
 public:
  MooncakeKVCacheTransferDefault(const int32_t device_id,
                                 const int16_t listen_port,
                                 const torch::Device& device,
                                 const std::string& model_type);

  void allocate_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      torch::ScalarType dtype) override;

  void register_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      const torch::ScalarType dtype) override;

  bool pull_kv_blocks(const uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const int64_t src_k_cache_id,
                      const int64_t src_v_cache_id,
                      const std::vector<uint64_t>& src_blocks,
                      const std::vector<uint64_t>& dst_blocks) override;

  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft) override;

 private:
  void allocate_kv_cache_impl(
      std::vector<xllm::KVCache>& kv_caches,
      int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      torch::ScalarType dtype);

  void add_buf(const torch::Tensor& tensor,
               std::vector<void*>& addrs,
               std::vector<size_t>& lens,
               std::vector<uint64_t>& buf_bytes) const;
  std::vector<int64_t> get_buf_ids(const std::vector<int64_t>& layer_ids) const;

  // Register per-layer K/V tensor memory.
  void register_kv_cache_impl(std::vector<xllm::KVCache>& kv_caches);

  bool has_v_cache_ = true;
  bool has_index_cache_ = false;
  int64_t buf_cnt_per_layer_ = 2;
  std::string model_type_;
};

class MooncakeKVCacheTransferXTensor final
    : public MooncakeKVCacheTransferBase {
 public:
  MooncakeKVCacheTransferXTensor(const int32_t device_id,
                                 const int16_t listen_port,
                                 const torch::Device& device);

  void set_model_id(const std::string& model_id) { model_id_ = model_id; }

  void allocate_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      torch::ScalarType dtype) override;

  void register_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      const torch::ScalarType dtype) override;

  bool pull_kv_blocks(const uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const int64_t src_k_cache_id,
                      const int64_t src_v_cache_id,
                      const std::vector<uint64_t>& src_blocks,
                      const std::vector<uint64_t>& dst_blocks) override;

  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft) override;

 private:
  void allocate_kv_cache_impl(
      std::vector<xllm::KVCache>& kv_caches,
      int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      torch::ScalarType dtype);

  // Register GlobalXTensor memory region.
  void register_kv_cache_impl();

  bool pull_kv_blocks_impl(const std::string& src_addr,
                           const std::vector<uint64_t>& src_blocks,
                           const std::vector<uint64_t>& dst_blocks);

  bool push_kv_blocks_impl(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer);

  std::string model_id_;
};

}  // namespace xllm
