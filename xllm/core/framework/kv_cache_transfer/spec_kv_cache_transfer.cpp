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

#include "framework/kv_cache_transfer/spec_kv_cache_transfer.h"

#include <glog/logging.h>

#include "common/macros.h"

namespace xllm {
SpecKVCacheTransfer::SpecKVCacheTransfer(const std::string& device_ip,
                                         const uint16_t listen_port,
                                         const InstanceRole& instance_role,
                                         const std::string& model_type)
    : LlmDataDistTransfer(device_ip, listen_port, instance_role, model_type) {}

void SpecKVCacheTransfer::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  register_kv_cache_internal(kv_caches, layer_registered_caches_);
}

void SpecKVCacheTransfer::register_kv_cache_spec(
    std::vector<xllm::KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype) {
  UNUSED_PARAMETER(kv_cache_shape);
  UNUSED_PARAMETER(dtype);
  register_kv_cache_internal(kv_caches, spec_layer_registered_caches_);
}

void SpecKVCacheTransfer::register_kv_cache_internal(
    std::vector<xllm::KVCache>& kv_caches,
    LayerRegisteredCaches& layer_registered_caches) {
  register_layer_registered_caches(kv_caches, layer_registered_caches);
}

void SpecKVCacheTransfer::free_kv_cache() {
  layer_registered_caches_.clear();
  spec_layer_registered_caches_.clear();
}

bool SpecKVCacheTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  const bool base_success = LlmDataDistTransfer::pull_kv_blocks(src_cluster_id,
                                                                src_addr,
                                                                src_k_cache_id,
                                                                src_v_cache_id,
                                                                src_blocks,
                                                                dst_blocks);
  bool spec_success = true;
  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(spec_layer_registered_caches_.size());
       ++layer_id) {
    const auto& registered_caches = spec_layer_registered_caches_[layer_id];
    for (const RegisteredCache& registered_cache : registered_caches) {
      CacheIndex cache_index{src_cluster_id, registered_cache.cache.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range = {0, 0};
      ext_param.dst_layer_range = {0, 0};
      ext_param.tensor_num_per_layer = 1;
      auto ret = llm_data_dist_->PullKvBlocks(cache_index,
                                              registered_cache.cache,
                                              src_blocks,
                                              dst_blocks,
                                              ext_param);
      if (ret != LLM_SUCCESS) {
        LOG(ERROR) << "Pull spec KvBlocks failed, layer = " << layer_id
                   << ", ret = " << std::hex << ret;
        spec_success = false;
      }
    }
  }
  return base_success && spec_success;
}

bool SpecKVCacheTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  if (is_spec_draft) {
    return push_kv_blocks_spec(merged_kv_infos, layer_synchronizer);
  } else {
    return push_kv_blocks_internal(
        merged_kv_infos, layer_synchronizer, layer_registered_caches_);
  }
}

bool SpecKVCacheTransfer::push_kv_blocks_spec(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer) {
  return push_kv_blocks_internal(
      merged_kv_infos, layer_synchronizer, spec_layer_registered_caches_);
}

bool SpecKVCacheTransfer::push_kv_blocks_internal(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    const LayerRegisteredCaches& layer_registered_caches) {
  return push_layer_registered_caches(
      layer_registered_caches, merged_kv_infos, layer_synchronizer);
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
