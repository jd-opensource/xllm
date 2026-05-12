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

#include "framework/kv_cache_transfer/llm_data_dist_transfer.h"

#include <glog/logging.h>

#include "core/framework/config/xllm_config.h"
#include "util/net.h"

namespace xllm {

const std::map<torch::ScalarType, ge::DataType> kScalarTypeToDtype = {
    {torch::kBool, ge::DT_BOOL},
    {torch::kByte, ge::DT_UINT8},
    {torch::kChar, ge::DT_INT8},
    {torch::kShort, ge::DT_INT16},
    {torch::kInt, ge::DT_INT32},
    {torch::kLong, ge::DT_INT64},
    {torch::kBFloat16, ge::DT_BF16},
    {torch::kHalf, ge::DT_FLOAT16},
    {torch::kFloat, ge::DT_FLOAT},
    {torch::kDouble, ge::DT_DOUBLE},
};

LlmDataDistTransfer::LlmDataDistTransfer(const std::string& device_ip,
                                         const uint16_t listen_port,
                                         const InstanceRole& instance_role,
                                         const std::string& model_type,
                                         bool enable_lighting_indexer)
    : device_ip_(device_ip),
      listen_port_(listen_port),
      enable_lighting_indexer_(enable_lighting_indexer),
      model_type_(model_type),
      KVCacheTransfer() {
  LlmRole role;
  if (instance_role == InstanceRole::PREFILL) {
    LOG(INFO) << "Create LlmDataDistTransfer for prefill instance.";
    role = LlmRole::kPrompt;
  } else if (instance_role == InstanceRole::DECODE) {
    LOG(INFO) << "Create LlmDataDistTransfer for decode instance.";
    role = LlmRole::kDecoder;
  } else {
    LOG(INFO) << "Create LlmDataDistTransfer for mix instance.";
    role = LlmRole::kMix;
  }
  host_ip_ = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(host_ip_, listen_port);
  llm_data_dist_ = std::make_shared<LlmDataDist>(cluster_id_, role);
}

void LlmDataDistTransfer::initialize(int32_t device_id) {
  std::map<AscendString, AscendString> options;
  options[OPTION_DEVICE_ID] = std::to_string(device_id).c_str();

  std::string local_ip_info = host_ip_ + ":" + std::to_string(listen_port_);
  options[OPTION_LISTEN_IP_INFO] = local_ip_info.c_str();

  auto ret = llm_data_dist_->Initialize(options);
  CHECK(ret == LLM_SUCCESS)
      << "Initialize LlmDataList failed, ret = " << std::hex << ret;
  LOG(INFO) << "Initialize LlmDataList success.";
}

void LlmDataDistTransfer::finalize() { llm_data_dist_->Finalize(); }

void LlmDataDistTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  CHECK(!kv_caches.empty()) << "KV caches must be allocated before register.";
  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  num_layers_ = num_layers;
  const std::vector<int64_t>& key_cache_shape =
      kv_cache_shape.key_cache_shape();
  const std::vector<int64_t>& value_cache_shape =
      kv_cache_shape.value_cache_shape();
  const std::vector<int64_t>& index_cache_shape =
      kv_cache_shape.index_cache_shape();

  const auto& it = kScalarTypeToDtype.find(dtype);
  CHECK(it != kScalarTypeToDtype.cend()) << "Unsupport data type : " << dtype;
  auto ge_dtype = it->second;

  if (enable_lighting_indexer_) {
    CHECK(kv_cache_shape.has_index_cache_shape())
        << "index_cache_shape is required when lighting indexer is enabled.";
  }

  std::vector<uint64_t> k_cache_addrs;
  std::vector<uint64_t> v_cache_addrs;
  std::vector<uint64_t> index_cache_addrs;
  k_cache_.tensor_addrs.clear();
  v_cache_.tensor_addrs.clear();
  index_cache_.tensor_addrs.clear();
  k_cache_addrs.reserve(num_layers);
  v_cache_addrs.reserve(num_layers);
  k_cache_.tensor_addrs.reserve(num_layers);
  v_cache_.tensor_addrs.reserve(num_layers);
  if (enable_lighting_indexer_) {
    index_cache_addrs.reserve(num_layers);
    index_cache_.tensor_addrs.reserve(num_layers);
  }

  for (int64_t i = 0; i < num_layers; ++i) {
    torch::Tensor key_cache = kv_caches[i].get_k_cache();
    torch::Tensor value_cache = kv_caches[i].get_v_cache();
    CHECK(key_cache.defined() && key_cache.numel() > 0)
        << "key cache is not allocated at layer " << i;
    CHECK(value_cache.defined() && value_cache.numel() > 0)
        << "value cache is not allocated at layer " << i;

    void* k_cache_buffer = key_cache.data_ptr();
    void* v_cache_buffer = value_cache.data_ptr();
    k_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(k_cache_buffer));
    v_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(v_cache_buffer));
    k_cache_.tensor_addrs.emplace_back(
        reinterpret_cast<uintptr_t>(k_cache_buffer));
    v_cache_.tensor_addrs.emplace_back(
        reinterpret_cast<uintptr_t>(v_cache_buffer));

    if (enable_lighting_indexer_) {
      torch::Tensor index_cache = kv_caches[i].get_index_cache();
      CHECK(index_cache.defined() && index_cache.numel() > 0)
          << "index cache is not allocated at layer " << i;
      void* index_cache_buffer = index_cache.data_ptr();
      index_cache_addrs.emplace_back(
          reinterpret_cast<uint64_t>(index_cache_buffer));
      index_cache_.tensor_addrs.emplace_back(
          reinterpret_cast<uintptr_t>(index_cache_buffer));
    }
  }

  // register key cache
  CacheDesc& k_cache_desc = k_cache_.cache_desc;
  k_cache_desc.num_tensors = num_layers;
  k_cache_desc.data_type = ge_dtype;
  k_cache_desc.shape = key_cache_shape;
  auto ret = llm_data_dist_->RegisterKvCache(
      k_cache_desc, k_cache_addrs, {}, k_cache_.cache_id);
  CHECK(ret == LLM_SUCCESS)
      << "Register key cache failed, ret = " << std::hex << ret;

  // register value cache
  CacheDesc& v_cache_desc = v_cache_.cache_desc;
  v_cache_desc.num_tensors = num_layers;
  v_cache_desc.data_type = ge_dtype;
  v_cache_desc.shape = value_cache_shape;
  ret = llm_data_dist_->RegisterKvCache(
      v_cache_desc, v_cache_addrs, {}, v_cache_.cache_id);
  CHECK(ret == LLM_SUCCESS)
      << "Register value cache failed, ret = " << std::hex << ret;

  LOG(INFO) << "Register KV cache success.";

  if (enable_lighting_indexer_) {
    // register index cache
    CacheDesc& index_cache_desc = index_cache_.cache_desc;
    index_cache_desc.num_tensors = num_layers;
    index_cache_desc.data_type = ge_dtype;
    index_cache_desc.shape = index_cache_shape;
    ret = llm_data_dist_->RegisterKvCache(
        index_cache_desc, index_cache_addrs, {}, index_cache_.cache_id);
    CHECK(ret == LLM_SUCCESS)
        << "Register index cache failed, ret = " << std::hex << ret;
  }
}

void LlmDataDistTransfer::free_kv_cache() {
  k_cache_.tensor_addrs.clear();
  v_cache_.tensor_addrs.clear();
  index_cache_.tensor_addrs.clear();
}

void LlmDataDistTransfer::get_cache_info(uint64_t& cluster_id,
                                         std::string& addr,
                                         int64_t& key_cache_id,
                                         int64_t& value_cache_id) {
  cluster_id = cluster_id_;
  key_cache_id = k_cache_.cache_id;
  value_cache_id = v_cache_.cache_id;
}

bool LlmDataDistTransfer::link_cluster(const uint64_t cluster_id,
                                       const std::string& remote_addr,
                                       const std::string& device_ip,
                                       const uint16_t port) {
  if (linked_cluster_ids.find(cluster_id) != linked_cluster_ids.end()) {
    // The cluster is connected.
    return true;
  }

  std::vector<llm_datadist::Status> rets;
  std::vector<ClusterInfo> clusters;
  ClusterInfo cluster_info = create_cluster_info(cluster_id, device_ip, port);
  clusters.emplace_back(std::move(cluster_info));

  auto ret = llm_data_dist_->LinkLlmClusters(
      clusters, rets, /*timeout_in_millis=*/60000);
  if (ret != LLM_SUCCESS) {
    LOG(ERROR) << "LinkLlmClusters failed, ret = " << std::hex << ret;
    return false;
  }
  LOG(INFO) << "LinkLlmClusters success, ip : " << device_ip
            << ", port : " << port;
  linked_cluster_ids.insert(cluster_id);

  return true;
}

bool LlmDataDistTransfer::unlink_cluster(const uint64_t& cluster_id,
                                         const std::string& remote_addr,
                                         const std::string& remote_ip,
                                         const uint16_t remote_port,
                                         bool force_flag) {
  std::vector<llm_datadist::Status> rets;
  std::vector<ClusterInfo> clusters;
  ClusterInfo cluster_info =
      create_cluster_info(cluster_id, remote_ip, remote_port);
  clusters.emplace_back(std::move(cluster_info));

  auto ret =
      llm_data_dist_->UnlinkLlmClusters(clusters, rets, 1000, force_flag);
  if (ret != LLM_SUCCESS) {
    LOG(ERROR) << "UnlinkLlmClusters failed, ret = " << std::hex << ret;
    return false;
  }
  LOG(INFO) << "UnlinkLlmClusters success, ip : " << remote_ip
            << ", port : " << remote_port;
  linked_cluster_ids.erase(cluster_id);

  return true;
}

bool LlmDataDistTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  CacheIndex k_cache_index{src_cluster_id, src_k_cache_id};
  CacheIndex v_cache_index{src_cluster_id, src_v_cache_id};
  auto k_ret = llm_data_dist_->PullKvBlocks(
      k_cache_index, k_cache_, src_blocks, dst_blocks);
  auto v_ret = llm_data_dist_->PullKvBlocks(
      v_cache_index, v_cache_, src_blocks, dst_blocks);
  if (k_ret != LLM_SUCCESS || v_ret != LLM_SUCCESS) {
    LOG(ERROR) << "PullKvBlocks failed, k_ret = " << std::hex << k_ret
               << ", v_ret = " << std::hex << v_ret;
    return false;
  }
  return true;
}

bool LlmDataDistTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  bool result = true;
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);
    // Push the KV Cache computed at this layer for all requests to the
    // designated worker.
    for (const auto& pair : merged_kv_infos) {
      const KVCacheInfo& kv_info = pair.second;
      if (kv_info.src_blocks.size() == 0) {
        continue;
      }

      CacheIndex k_cache_index{kv_info.dst_cluster_id, kv_info.dst_k_cache_id};
      CacheIndex v_cache_index{kv_info.dst_cluster_id, kv_info.dst_v_cache_id};
      CacheIndex index_cache_index{kv_info.dst_cluster_id,
                                   index_cache_.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range =
          std::pair<int32_t, int32_t>(layer_index, layer_index);
      ext_param.dst_layer_range =
          std::pair<int32_t, int32_t>(layer_index, layer_index);
      ext_param.tensor_num_per_layer = 1;

      auto k_ret = llm_data_dist_->PushKvBlocks(k_cache_,
                                                k_cache_index,
                                                kv_info.src_blocks,
                                                kv_info.dst_blocks,
                                                ext_param);
      auto v_ret = llm_data_dist_->PushKvBlocks(v_cache_,
                                                v_cache_index,
                                                kv_info.src_blocks,
                                                kv_info.dst_blocks,
                                                ext_param);
      if (k_ret != LLM_SUCCESS || v_ret != LLM_SUCCESS) {
        LOG(ERROR) << "PushKvBlocks failed, layer = " << layer_index
                   << ", k_ret = " << std::hex << k_ret
                   << ", v_ret = " << std::hex << v_ret;
        result = false;
      }
      if (enable_lighting_indexer_) {
        auto index_ret = llm_data_dist_->PushKvBlocks(index_cache_,
                                                      index_cache_index,
                                                      kv_info.src_blocks,
                                                      kv_info.dst_blocks,
                                                      ext_param);
        if (index_ret != LLM_SUCCESS) {
          LOG(ERROR) << "PushKvBlocks failed, layer = " << layer_index
                     << ", index_ret = " << std::hex << index_ret;
          result = false;
        }
      }
    }
  }
  return result;
}

ClusterInfo LlmDataDistTransfer::create_cluster_info(
    const uint64_t& cluster_id,
    const std::string& remote_ip,
    const uint16_t& remote_port) {
  ClusterInfo cluster_info;
  IpInfo local_ip_info;
  IpInfo remote_ip_info;

  local_ip_info.ip = host_ip_.c_str();
  local_ip_info.port = listen_port_;
  remote_ip_info.ip = remote_ip.c_str();
  remote_ip_info.port = remote_port;
  cluster_info.remote_cluster_id = cluster_id;
  cluster_info.local_ip_infos.emplace_back(std::move(local_ip_info));
  cluster_info.remote_ip_infos.emplace_back(std::move(remote_ip_info));

  return cluster_info;
}

}  // namespace xllm
