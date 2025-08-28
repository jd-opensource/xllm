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

#include "llm_data_dist_transfer.h"

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

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
                                         const InstanceRole& instance_role)
    : device_ip_(device_ip), listen_port_(listen_port), KVCacheTransfer() {
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

  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port);
  llm_data_dist_ = std::make_shared<LlmDataDist>(cluster_id_, role);
}

void LlmDataDistTransfer::initialize(int32_t device_id,
                                     uint64_t buf_pool_size) {
  std::map<AscendString, AscendString> options;
  options[OPTION_DEVICE_ID] = std::to_string(device_id).c_str();

  std::string local_ip_info = device_ip_ + ":" + std::to_string(listen_port_);
  options[OPTION_LISTEN_IP_INFO] = local_ip_info.c_str();

  // TODO: Set all parameters in buf_cfg to option.
  std::ostringstream oss;
  oss << R"({"buf_cfg":[{"total_size":)" << 2097152 << R"(,"blk_size":)" << 256
      << R"(,"max_buf_size":)" << 8192 << R"(}],"buf_pool_size":)"
      << buf_pool_size << "}";
  options[OPTION_BUF_POOL_CFG] = oss.str().c_str();
  options[OPTION_ENABLE_SET_ROLE] = "1";
  auto ret = llm_data_dist_->Initialize(options);
  CHECK(ret == LLM_SUCCESS) << "Initialize LlmDataList failed, ret = " << ret;
  LOG(INFO) << "Initialize LlmDataList success.";
}

void LlmDataDistTransfer::finalize() { llm_data_dist_->Finalize(); }

void LlmDataDistTransfer::allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = num_layers;

  const auto& it = kScalarTypeToDtype.find(dtype);
  CHECK(it != kScalarTypeToDtype.cend()) << "Unsupport data type : " << dtype;
  auto ge_dtype = it->second;
  CacheDesc k_cache_desc;
  k_cache_desc.num_tensors = num_layers;
  k_cache_desc.data_type = ge_dtype;
  k_cache_desc.shape = kv_cache_shape[0];
  auto ret = llm_data_dist_->AllocateCache(k_cache_desc, k_cache_);
  CHECK(ret == LLM_SUCCESS)
      << "Allocate key cache failed, ret = " << std::hex << ret;

  CacheDesc v_cache_desc;
  v_cache_desc.num_tensors = num_layers;
  v_cache_desc.data_type = ge_dtype;
  v_cache_desc.shape = kv_cache_shape[1];
  ret = llm_data_dist_->AllocateCache(v_cache_desc, v_cache_);
  CHECK(ret == LLM_SUCCESS)
      << "Allocate value cache failed, ret = " << std::hex << ret;

  auto k_torch_tensors =
      convert_to_torch_tensor(kv_cache_shape[0], dtype, k_cache_.tensor_addrs);
  auto v_torch_tensors =
      convert_to_torch_tensor(kv_cache_shape[1], dtype, v_cache_.tensor_addrs);
  torch::Tensor key_cache, value_cache;
  for (int64_t i = 0; i < num_layers; ++i) {
    key_cache = k_torch_tensors[i];
    value_cache = v_torch_tensors[i];
    kv_caches.emplace_back(key_cache, value_cache);
  }
}

void LlmDataDistTransfer::free_kv_cache() {
  llm_data_dist_->DeallocateCache(k_cache_.cache_id);
  llm_data_dist_->DeallocateCache(v_cache_.cache_id);
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

  auto ret = llm_data_dist_->LinkLlmClusters(clusters, rets);
  if (ret != LLM_SUCCESS) {
    LOG(ERROR) << "LinkLlmClusters failed, ret = " << ret;
    return false;
  }
  LOG(INFO) << "LinkLlmClusters success.";
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
    LOG(ERROR) << "UnlinkLlmClusters failed, ret = " << ret;
    return false;
  }
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
    LOG(ERROR) << "PullKvBlocks failed, k_ret = " << k_ret
               << ", v_ret = " << v_ret;
    return false;
  }
  return true;
}

bool LlmDataDistTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
#if defined(USE_A3)
  LOG(FATAL) << "A3 does not support llmdatadist.";
  return false;
#else
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);
    // Push the KV Cache computed at this layer for all requests to the
    // designated worker.
    for (const auto& pair : merged_kv_infos) {
      const KVCacheInfo& kv_info = pair.second;
      CacheIndex k_cache_index{kv_info.dst_cluster_id, kv_info.dst_k_cache_id};
      CacheIndex v_cache_index{kv_info.dst_cluster_id, kv_info.dst_v_cache_id};
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
                   << ", k_ret = " << k_ret << ", v_ret = " << v_ret;
        return false;
      }
    }
  }
  return true;
#endif
}

std::vector<torch::Tensor> LlmDataDistTransfer::convert_to_torch_tensor(
    const std::vector<int64_t>& dims,
    const torch::ScalarType dtype,
    const std::vector<uintptr_t>& addresses) {
  std::vector<torch::Tensor> torch_tensors;
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

  torch_tensors.reserve(addresses.size());
  for (auto dev_addr : addresses) {
    auto tensor = torch::empty({0}, option);
    auto address = reinterpret_cast<void*>(dev_addr);
    torch::DataPtr c10_data_ptr(
        address, address, [](void*) {}, tensor.device());

    size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
        dims, tensor.dtype().itemsize());
    torch::Storage storage;
    // get npu storage constructor from register and construct storage
    auto fptr = c10::GetStorageImplCreate(device_type);
    auto allocator = c10::GetAllocator(device_type);
    storage = fptr(c10::StorageImpl::use_byte_size_t(), 0, allocator, true);
    storage.unsafeGetStorageImpl()->set_nbytes(tensor_nbytes);
    storage.set_data_ptr(std::move(c10_data_ptr));

    tensor.set_(storage, 0, dims);
    // cast npu format to nd
    tensor = at_npu::native::npu_format_cast(tensor, 2);
    torch_tensors.emplace_back(std::move(tensor));
  }
  return torch_tensors;
}

ClusterInfo LlmDataDistTransfer::create_cluster_info(
    const uint64_t& cluster_id,
    const std::string& remote_ip,
    const uint16_t& remote_port) {
  ClusterInfo cluster_info;
  IpInfo local_ip_info;
  IpInfo remote_ip_info;

  local_ip_info.ip = device_ip_.c_str();
  remote_ip_info.ip = remote_ip.c_str();
  remote_ip_info.port = remote_port;
  cluster_info.remote_cluster_id = cluster_id;
  cluster_info.local_ip_infos.emplace_back(std::move(local_ip_info));
  cluster_info.remote_ip_infos.emplace_back(std::move(remote_ip_info));

  return cluster_info;
}

}  // namespace xllm
