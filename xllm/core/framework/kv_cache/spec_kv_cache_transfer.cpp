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

#include "spec_kv_cache_transfer.h"

#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

namespace xllm {
namespace {
#define CHECK_LDD_RET(ret)  \
  CHECK(ret == LLM_SUCCESS) \
      << "Call LlmDataDist function failed, ret = " << std::hex << ret

#define CHECK_ACL_RET(ret)  \
  CHECK(ret == ACL_SUCCESS) \
      << "Call ACL function failed, ret = " << std::hex << ret

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
}  // namespace

SpecKVCacheTransfer::SpecKVCacheTransfer(const std::string& device_ip,
                                         const uint16_t listen_port,
                                         const InstanceRole& instance_role)
    : LlmDataDistTransfer(device_ip, listen_port, instance_role) {}

void SpecKVCacheTransfer::allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  allocate_kv_cache_internal(kv_caches,
                             num_layers,
                             kv_cache_shape,
                             dtype,
                             /*is_spec*/ false,
                             k_cache_,
                             v_cache_);
}

void SpecKVCacheTransfer::allocate_kv_cache_spec(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  allocate_kv_cache_internal(kv_caches,
                             num_layers,
                             kv_cache_shape,
                             dtype,
                             /*is_spec*/ true,
                             spec_k_cache_,
                             spec_v_cache_);
}
void SpecKVCacheTransfer::allocate_kv_cache_internal(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype,
    bool is_spec,
    Cache& k_cache,
    Cache& v_cache) {
  if (is_spec) {
    spec_num_layers_ = num_layers;
  } else {
    num_layers_ = num_layers;
  }

  const auto& it = kScalarTypeToDtype.find(dtype);
  CHECK(it != kScalarTypeToDtype.cend()) << "Unsupport data type : " << dtype;
  auto ge_dtype = it->second;

  // calculate the size of kv cache for each layer
  auto data_size = torch::elementSize(dtype);
  int64_t k_cache_size_per_layer = std::accumulate(kv_cache_shape[0].begin(),
                                                   kv_cache_shape[0].end(),
                                                   data_size,
                                                   std::multiplies<int64_t>());
  int64_t v_cache_size_per_layer = std::accumulate(kv_cache_shape[1].begin(),
                                                   kv_cache_shape[1].end(),
                                                   data_size,
                                                   std::multiplies<int64_t>());

  // allocate device memory for kv cache
  std::vector<uint64_t> k_cache_addrs;
  std::vector<uint64_t> v_cache_addrs;
  k_cache_addrs.reserve(num_layers);
  v_cache_addrs.reserve(num_layers);
  k_cache.tensor_addrs.reserve(num_layers);
  v_cache.tensor_addrs.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    void* k_cache_buffer = nullptr;
    void* v_cache_buffer = nullptr;
    CHECK_ACL_RET(aclrtMalloc(
        &k_cache_buffer, k_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY));
    CHECK_ACL_RET(aclrtMalloc(
        &v_cache_buffer, v_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY));

    k_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(k_cache_buffer));
    v_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(v_cache_buffer));
    k_cache.tensor_addrs.emplace_back(
        reinterpret_cast<uintptr_t>(k_cache_buffer));
    v_cache.tensor_addrs.emplace_back(
        reinterpret_cast<uintptr_t>(v_cache_buffer));
  }

  // convert memory addrs to torch tensors
  auto k_torch_tensors =
      convert_to_torch_tensor(kv_cache_shape[0], dtype, k_cache.tensor_addrs);
  auto v_torch_tensors =
      convert_to_torch_tensor(kv_cache_shape[1], dtype, v_cache.tensor_addrs);
  torch::Tensor key_cache, value_cache;
  for (int64_t i = 0; i < num_layers; ++i) {
    key_cache = k_torch_tensors[i];
    value_cache = v_torch_tensors[i];
    kv_caches.emplace_back(key_cache, value_cache);
  }

  // register key cache
  CacheDesc& k_cache_desc = k_cache.cache_desc;
  k_cache_desc.num_tensors = num_layers;
  k_cache_desc.data_type = ge_dtype;
  k_cache_desc.shape = kv_cache_shape[0];
  auto ret = llm_data_dist_->RegisterKvCache(
      k_cache_desc, k_cache_addrs, {}, k_cache.cache_id);
  CHECK(ret == LLM_SUCCESS)
      << "Register key cache failed, ret = " << std::hex << ret;

  // register value cache
  CacheDesc& v_cache_desc = v_cache.cache_desc;
  v_cache_desc.num_tensors = num_layers;
  v_cache_desc.data_type = ge_dtype;
  v_cache_desc.shape = kv_cache_shape[1];
  ret = llm_data_dist_->RegisterKvCache(
      v_cache_desc, v_cache_addrs, {}, v_cache.cache_id);
  CHECK(ret == LLM_SUCCESS)
      << "Register value cache failed, ret = " << std::hex << ret;

  LOG(INFO) << "Register KV cache success.";
}

void SpecKVCacheTransfer::free_kv_cache() {
  auto free_cache = [](const std::vector<uintptr_t>& tensor_addrs) {
    for (auto tensor_addr : tensor_addrs) {
      CHECK_ACL_RET(aclrtFree(reinterpret_cast<void*>(tensor_addr)));
    }
  };
  free_cache(k_cache_.tensor_addrs);
  free_cache(v_cache_.tensor_addrs);
  free_cache(spec_k_cache_.tensor_addrs);
  free_cache(spec_v_cache_.tensor_addrs);
}

bool SpecKVCacheTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  CacheIndex k_cache_index{src_cluster_id, src_k_cache_id};
  CHECK_LDD_RET(llm_data_dist_->PullKvBlocks(
      k_cache_index, k_cache_, src_blocks, dst_blocks));
  CacheIndex v_cache_index{src_cluster_id, src_v_cache_id};
  CHECK_LDD_RET(llm_data_dist_->PullKvBlocks(
      v_cache_index, v_cache_, src_blocks, dst_blocks));

  CacheIndex spec_k_cache_index{src_cluster_id, spec_k_cache_.cache_id};
  CHECK_LDD_RET(llm_data_dist_->PullKvBlocks(
      spec_k_cache_index, spec_k_cache_, src_blocks, dst_blocks));
  CacheIndex spec_v_cache_index{src_cluster_id, spec_v_cache_.cache_id};
  CHECK_LDD_RET(llm_data_dist_->PullKvBlocks(
      spec_v_cache_index, spec_v_cache_, src_blocks, dst_blocks));

  return true;
}

bool SpecKVCacheTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  if (is_spec_draft) {
    return push_kv_blocks_spec(merged_kv_infos, layer_synchronizer);
  } else {
    return push_kv_blocks_internal(
        merged_kv_infos, layer_synchronizer, num_layers_, k_cache_, v_cache_);
  }
}

bool SpecKVCacheTransfer::push_kv_blocks_spec(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer) {
  return push_kv_blocks_internal(merged_kv_infos,
                                 layer_synchronizer,
                                 spec_num_layers_,
                                 spec_k_cache_,
                                 spec_v_cache_);
}

bool SpecKVCacheTransfer::push_kv_blocks_internal(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    int64_t num_layers,
    const Cache& k_cache,
    const Cache& v_cache) {
  for (int64_t layer_index = 0; layer_index < num_layers; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);

    // Push the KV Cache computed at this layer for all requests to the
    // designated worker.
    for (const auto& pair : merged_kv_infos) {
      const KVCacheInfo& kv_info = pair.second;
      CacheIndex k_cache_index{kv_info.dst_cluster_id, k_cache.cache_id};
      CacheIndex v_cache_index{kv_info.dst_cluster_id, v_cache.cache_id};

      KvCacheExtParam ext_param{};
      ext_param.src_layer_range = {layer_index, layer_index};
      ext_param.dst_layer_range = {layer_index, layer_index};
      ext_param.tensor_num_per_layer = 1;

      CHECK_LDD_RET(llm_data_dist_->PushKvBlocks(k_cache,
                                                 k_cache_index,
                                                 kv_info.src_blocks,
                                                 kv_info.dst_blocks,
                                                 ext_param));
      CHECK_LDD_RET(llm_data_dist_->PushKvBlocks(v_cache,
                                                 v_cache_index,
                                                 kv_info.src_blocks,
                                                 kv_info.dst_blocks,
                                                 ext_param));
    }
  }
  return true;
}

folly::SemiFuture<bool> SpecKVCacheTransfer::push_kv_blocks_async(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args,
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer,
    bool is_spec_draft) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        &transfer_kv_infos,
                        &parallel_args,
                        layer_synchronizer,
                        is_spec_draft,
                        promise = std::move(promise)]() mutable {
    std::unordered_map<std::string, KVCacheInfo> merged_kv_infos;
    merge_kv_blocks(merged_kv_infos, transfer_kv_infos, parallel_args);
    bool success = true;
    if (!merged_kv_infos.empty()) {
      success = this->push_kv_blocks(
          merged_kv_infos, layer_synchronizer, is_spec_draft);
    }
    promise.setValue(success);
  });
  return future;
}

void SpecKVCacheTransfer::merge_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args) {
  // Obtain the parallel parameters of the source instance
  int32_t src_rank = parallel_args.rank();
  int32_t src_dp_size = parallel_args.dp_size();
  int32_t src_world_size = parallel_args.world_size();
  int32_t src_tp_size = src_world_size / src_dp_size;
  int32_t src_dp_local_tp_rank = src_rank % src_tp_size;
  for (auto& info : transfer_kv_infos) {
    // Obtain the parallel parameters of the destination instance.
    int32_t dst_dp_rank = info.dp_rank;
    int32_t dst_dp_size = info.remote_instance_info.dp_size;
    int32_t dst_world_size = info.remote_instance_info.cluster_ids.size();
    int32_t dst_tp_size = dst_world_size / dst_dp_size;
    // Get the DP groups of the destination instance connected to the current
    // worker.
    std::unordered_set<int32_t> linked_dp_ranks;
    for (int32_t i = src_dp_local_tp_rank; i < dst_world_size;
         i += src_tp_size) {
      int32_t linked_dp_rank = i / dst_tp_size;
      linked_dp_ranks.emplace(linked_dp_rank);
    }
    // If the target DP rank of the request is not linked to the current worker,
    // skip the request.
    if (linked_dp_ranks.find(dst_dp_rank) == linked_dp_ranks.end()) {
      continue;
    }
    // The current worker needs to push the KV Cache to all workers in the
    // destination DP group it is connected to.
    for (int32_t i =
             src_dp_local_tp_rank % dst_tp_size + dst_tp_size * dst_dp_rank;
         i < dst_tp_size * (dst_dp_rank + 1);
         i += src_tp_size) {
      uint64_t dst_cluster_id = info.remote_instance_info.cluster_ids[i];
      auto& dst_addr = info.remote_instance_info.addrs[i];
      int64_t k_cache_id = info.remote_instance_info.k_cache_ids[i];
      int64_t v_cache_id = info.remote_instance_info.v_cache_ids[i];
      std::string key = std::to_string(dst_cluster_id) + "_" + dst_addr + "_" +
                        std::to_string(k_cache_id) + "_" +
                        std::to_string(v_cache_id);
      // Merge all kv blocks with the same destination worker into a single
      // vector.
      if (merged_kv_infos.find(key) == merged_kv_infos.end()) {
        KVCacheInfo kv_info;
        kv_info.dst_cluster_id = dst_cluster_id;
        kv_info.dst_addr = dst_addr;
        kv_info.dst_k_cache_id = k_cache_id;
        kv_info.dst_v_cache_id = v_cache_id;
        kv_info.src_blocks.insert(kv_info.src_blocks.end(),
                                  info.local_blocks_ids.begin(),
                                  info.local_blocks_ids.end());
        kv_info.dst_blocks.insert(kv_info.dst_blocks.end(),
                                  info.remote_blocks_ids.begin(),
                                  info.remote_blocks_ids.end());
        merged_kv_infos[key] = std::move(kv_info);
      } else {
        merged_kv_infos[key].src_blocks.insert(
            merged_kv_infos[key].src_blocks.end(),
            info.local_blocks_ids.begin(),
            info.local_blocks_ids.end());
        merged_kv_infos[key].dst_blocks.insert(
            merged_kv_infos[key].dst_blocks.end(),
            info.remote_blocks_ids.begin(),
            info.remote_blocks_ids.end());
      }
    }
  }
}
}  // namespace xllm
