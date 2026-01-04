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

#include "kv_cache_transfer.h"

#include <glog/logging.h>

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include "hccl_kv_cache_transfer.h"
#include "llm_data_dist_transfer.h"
#include "mooncake_kv_cache_transfer.h"
#endif

namespace xllm {

folly::SemiFuture<bool> KVCacheTransfer::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        src_cluster_id,
                        src_addr,
                        src_k_cache_id,
                        src_v_cache_id,
                        &src_blocks,
                        &dst_blocks,
                        promise = std::move(promise)]() mutable {
    const bool success = pull_kv_blocks(src_cluster_id,
                                        src_addr,
                                        src_k_cache_id,
                                        src_v_cache_id,
                                        src_blocks,
                                        dst_blocks);
    promise.setValue(success);
  });
  return future;
}

#if defined(USE_NPU)
folly::SemiFuture<bool> KVCacheTransfer::push_kv_blocks_async(
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
#endif

void KVCacheTransfer::merge_kv_blocks(
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

#if defined(USE_NPU)
std::vector<torch::Tensor> KVCacheTransfer::convert_to_torch_tensor(
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
#endif

std::shared_ptr<KVCacheTransfer> KVCacheTransferFactory::create(
    const std::string& transfer_type,
    const std::string& device_ip,
    uint16_t transfer_listen_port,
    InstanceRole instance_role,
    const Device& device,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype,
    std::vector<xllm::KVCache>& kv_caches,
    int64_t num_layers,
    std::function<void(const std::vector<std::vector<int64_t>>&)>
        allocate_kv_cache_func) {
  std::shared_ptr<KVCacheTransfer> transfer;

  int32_t device_id = device.index();

#if defined(USE_NPU)
  if (transfer_type == "LlmDataDist") {
    transfer = std::make_shared<LlmDataDistTransfer>(
        device_ip, transfer_listen_port, instance_role);

    kv_caches.reserve(num_layers);

    transfer->initialize(device_id);
    transfer->allocate_kv_cache(kv_caches, num_layers, kv_cache_shape, dtype);
  } else if (transfer_type == "Mooncake") {
    transfer = std::make_shared<MooncakeKVCacheTransfer>(
        device_id, transfer_listen_port, device);

    transfer->initialize(device_id);
    transfer->allocate_kv_cache(kv_caches, num_layers, kv_cache_shape, dtype);
    transfer->register_kv_cache(kv_caches, kv_cache_shape, dtype);
  } else {
    transfer =
        std::make_shared<HcclKVCacheTransfer>(device_id, transfer_listen_port);

    allocate_kv_cache_func(kv_cache_shape);
    transfer->register_kv_cache(kv_caches, kv_cache_shape, dtype);
  }
#endif

  return transfer;
}

}  // namespace xllm
