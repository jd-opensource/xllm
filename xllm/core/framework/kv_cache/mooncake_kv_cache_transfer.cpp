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

#include "mooncake_kv_cache_transfer.h"

#include <glog/logging.h>

#include "util/net.h"

namespace xllm {

MooncakeKVCacheTransfer::MooncakeKVCacheTransfer(const int32_t device_id,
                                                 const int16_t listen_port,
                                                 const torch::Device& device)
    : device_id_(device_id), listen_port_(listen_port), KVCacheTransfer() {
  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port_);

  mooncake_te_ = std::make_unique<MooncakeTransferEngine>(listen_port_, device);
}

void MooncakeKVCacheTransfer::initialize(int32_t device_id) {
  addr_ = mooncake_te_->initialize();
}

void MooncakeKVCacheTransfer::allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = num_layers;

  // calculate the size of kv cache for each layer
  auto data_size = torch::elementSize(dtype);
  int64_t k_cache_size_per_layer = data_size;
  for (int64_t i = 0; i < kv_cache_shape[0].size(); ++i) {
    k_cache_size_per_layer *= kv_cache_shape[0][i];
  }
  int64_t v_cache_size_per_layer = data_size;
  for (int64_t i = 0; i < kv_cache_shape[1].size(); ++i) {
    v_cache_size_per_layer *= kv_cache_shape[1][i];
  }

  // allocate device memory for kv cache
  std::vector<uint64_t> k_cache_addrs;
  std::vector<uint64_t> v_cache_addrs;
  k_cache_addrs.reserve(num_layers);
  v_cache_addrs.reserve(num_layers);

  std::vector<uintptr_t> k_tensor_addrs;
  std::vector<uintptr_t> v_tensor_addrs;
  k_tensor_addrs.reserve(num_layers);
  v_tensor_addrs.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    void* k_cache_buffer = nullptr;
    void* v_cache_buffer = nullptr;
    auto acl_ret = aclrtMalloc(
        &k_cache_buffer, k_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK(acl_ret == ACL_SUCCESS) << "aclrtMalloc k cache failed.";
    acl_ret = aclrtMalloc(
        &v_cache_buffer, v_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK(acl_ret == ACL_SUCCESS) << "aclrtMalloc v cache failed.";

    k_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(k_cache_buffer));
    v_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(v_cache_buffer));

    k_tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(k_cache_buffer));
    v_tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(v_cache_buffer));
  }

  // convert memory addrs to torch tensors
  auto k_torch_tensors =
      convert_to_torch_tensor(kv_cache_shape[0], dtype, k_tensor_addrs);
  auto v_torch_tensors =
      convert_to_torch_tensor(kv_cache_shape[1], dtype, v_tensor_addrs);

  torch::Tensor key_cache, value_cache;
  for (int64_t i = 0; i < num_layers; ++i) {
    key_cache = k_torch_tensors[i];
    value_cache = v_torch_tensors[i];
    kv_caches.emplace_back(key_cache, value_cache);
  }
}

void MooncakeKVCacheTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = kv_caches.size();
  int64_t num_cache = num_layers_ * 2;

  std::vector<void*> cache_addrs;
  std::vector<size_t> cache_lens;
  cache_addrs.reserve(num_cache);
  cache_lens.reserve(num_cache);

  for (int32_t i = 0; i < num_layers_; ++i) {
    cache_addrs.emplace_back(kv_caches[i].get_k_cache().data_ptr());
    cache_lens.emplace_back(kv_caches[i].get_k_cache().nbytes());
  }

  for (int32_t i = 0; i < num_layers_; ++i) {
    cache_addrs.emplace_back(kv_caches[i].get_v_cache().data_ptr());
    cache_lens.emplace_back(kv_caches[i].get_v_cache().nbytes());
  }

  int64_t data_size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  int64_t count_per_block = 1;
  for (int32_t i = 1; i < kv_cache_shape[0].size(); ++i) {
    count_per_block *= kv_cache_shape[0][i];
  }
  int64_t size_per_block = count_per_block * data_size;

  if (!mooncake_te_->register_memory(cache_addrs, cache_lens, size_per_block)) {
    LOG(ERROR) << "register_memory failed";
    return;
  }

  LOG(INFO) << "register_kv_cache success";
}

void MooncakeKVCacheTransfer::get_cache_info(uint64_t& cluster_id,
                                             std::string& addr,
                                             int64_t& key_cache_id,
                                             int64_t& value_cache_id) {
  cluster_id = cluster_id_;
  addr = addr_;
  key_cache_id = 0;
  value_cache_id = 0;

  LOG(INFO) << "get_cache_info success, cluster_id=" << cluster_id_
            << ", addr=" << addr_;
}

bool MooncakeKVCacheTransfer::link_cluster(const uint64_t cluster_id,
                                           const std::string& remote_addr,
                                           const std::string& device_ip,
                                           const uint16_t port) {
  LOG(INFO) << "link_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->open_session(cluster_id, remote_addr);
}

bool MooncakeKVCacheTransfer::unlink_cluster(const uint64_t& cluster_id,
                                             const std::string& remote_addr,
                                             const std::string& device_ip,
                                             const uint16_t port,
                                             bool force_flag) {
  LOG(INFO) << "unlink_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->close_session(cluster_id, remote_addr);
}

bool MooncakeKVCacheTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  std::vector<int64_t> layer_ids;
  auto ret = mooncake_te_->pull_memory_blocks(
      src_addr, src_blocks, dst_blocks, layer_ids);
  if (!ret) {
    LOG(ERROR) << "Pull kv cache blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

bool MooncakeKVCacheTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);
    // Push the KV Cache computed at this layer for all requests to the
    // designated worker.
    for (const auto& pair : merged_kv_infos) {
      std::vector<int64_t> layer_ids = {layer_index};
      const KVCacheInfo& kv_info = pair.second;
      auto ret = mooncake_te_->push_memory_blocks(
          kv_info.dst_addr, kv_info.src_blocks, kv_info.dst_blocks, layer_ids);
      if (!ret) {
        LOG(ERROR) << "Push kv blocks failed, layer = " << layer_index
                   << ", ret = " << ret;
        return false;
      }
    }
  }
  return true;
}

}  // namespace xllm
