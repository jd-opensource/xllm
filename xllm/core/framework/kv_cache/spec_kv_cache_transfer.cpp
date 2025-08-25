#include "spec_kv_cache_transfer.h"

#include <glog/logging.h>
#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#endif

namespace xllm {
#if defined(USE_NPU)
namespace {
#define CHECK_LDD_RET(ret)  \
  CHECK(ret == LLM_SUCCESS) \
      << "Call LlmDataDist function failed, ret = " << std::hex << ret

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
#endif

SpecKVCacheTransfer::SpecKVCacheTransfer(const std::string& device_ip,
                                         const uint16_t listen_port,
                                         const InstanceRole& instance_role)
    : LlmDataDistTransfer(device_ip, listen_port, instance_role) {}

void SpecKVCacheTransfer::allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  _allocate_kv_cache(kv_caches,
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
  _allocate_kv_cache(kv_caches,
                     num_layers,
                     kv_cache_shape,
                     dtype,
                     /*is_spec*/ true,
                     spec_k_cache_,
                     spec_v_cache_);
}
void SpecKVCacheTransfer::_allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype,
    bool is_spec,
    Cache& k_cache,
    Cache& v_cache) {
#if defined(USE_NPU)
  if (is_spec) {
    spec_num_layers_ = num_layers;
  } else {
    num_layers_ = num_layers;
  }

  const auto& it = kScalarTypeToDtype.find(dtype);
  CHECK(it != kScalarTypeToDtype.cend()) << "Unsupport data type : " << dtype;
  auto ge_dtype = it->second;
  CacheDesc k_cache_desc;
  k_cache_desc.num_tensors = num_layers;
  k_cache_desc.data_type = ge_dtype;
  k_cache_desc.shape = kv_cache_shape[0];
  CHECK_LDD_RET(llm_data_dist_->AllocateCache(k_cache_desc, k_cache));

  CacheDesc v_cache_desc;
  v_cache_desc.num_tensors = num_layers;
  v_cache_desc.data_type = ge_dtype;
  v_cache_desc.shape = kv_cache_shape[1];
  CHECK_LDD_RET(llm_data_dist_->AllocateCache(v_cache_desc, v_cache));

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
#endif
}

void SpecKVCacheTransfer::allocate_embedding(
    std::shared_ptr<EmbeddingAllocator> embedding_allocator,
    const std::vector<int64_t>& embedding_shape,
    torch::ScalarType dtype,
    torch::Device device) {
#if defined(USE_NPU)
  const auto& it = kScalarTypeToDtype.find(dtype);
  CHECK(it != kScalarTypeToDtype.cend()) << "Unsupport data type : " << dtype;
  auto ge_dtype = it->second;
  CacheDesc embed_cache_desc;
  embed_cache_desc.num_tensors = 1;
  embed_cache_desc.data_type = ge_dtype;
  embed_cache_desc.shape = embedding_shape;
  CHECK_LDD_RET(llm_data_dist_->AllocateCache(embed_cache_desc, embed_cache_));

  embed_host_cache_.cache_desc = embed_cache_.cache_desc;
  embed_host_cache_.cache_desc.placement = CachePlacement::kHost;
  CHECK_EQ(embed_host_cache_.cache_desc.num_tensors, 1);
  embed_host_cache_.tensor_addrs.emplace_back(reinterpret_cast<uint64_t>(
      embedding_allocator->get_embeddings_cache_ptr()));
#endif
}

void SpecKVCacheTransfer::free_kv_cache() {
#if defined(USE_NPU)
  llm_data_dist_->DeallocateCache(k_cache_.cache_id);
  llm_data_dist_->DeallocateCache(v_cache_.cache_id);
  llm_data_dist_->DeallocateCache(spec_k_cache_.cache_id);
  llm_data_dist_->DeallocateCache(spec_v_cache_.cache_id);
  llm_data_dist_->DeallocateCache(embed_cache_.cache_id);
#endif
}

bool SpecKVCacheTransfer::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
#if defined(USE_NPU)
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

  CacheIndex embed_cache_index{src_cluster_id, embed_cache_.cache_id};
  CHECK_LDD_RET(llm_data_dist_->PullKvBlocks(embed_cache_index,
                                             embed_cache_,
                                             {src_blocks.back()},
                                             {dst_blocks.back()}));
  return true;
#endif
}

#if defined(USE_NPU)
bool SpecKVCacheTransfer::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
#if defined(USE_A3)
  LOG(FATAL) << "A3 does not support llmdatadist.";
  return false;
#else
  if (!layer_synchronizer) {
    return push_embed_blocks(merged_kv_infos);
  }

  if (is_spec_draft) {
    return push_kv_blocks_spec(merged_kv_infos, layer_synchronizer);
  }
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);
    // Push the KV Cache computed at this layer for all requests to the
    // designated worker.
    for (const auto& pair : merged_kv_infos) {
      const KVCacheInfo& kv_info = pair.second;
      CacheIndex k_cache_index{kv_info.dst_cluster_id, k_cache_.cache_id};
      CacheIndex v_cache_index{kv_info.dst_cluster_id, v_cache_.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range =
          std::pair<int32_t, int32_t>(layer_index, layer_index);
      ext_param.dst_layer_range =
          std::pair<int32_t, int32_t>(layer_index, layer_index);
      ext_param.tensor_num_per_layer = 1;
      CHECK_LDD_RET(llm_data_dist_->PushKvBlocks(k_cache_,
                                                 k_cache_index,
                                                 kv_info.src_blocks,
                                                 kv_info.dst_blocks,
                                                 ext_param));
      CHECK_LDD_RET(llm_data_dist_->PushKvBlocks(v_cache_,
                                                 v_cache_index,
                                                 kv_info.src_blocks,
                                                 kv_info.dst_blocks,
                                                 ext_param));
    }
  }
  return true;
#endif
}

bool SpecKVCacheTransfer::push_kv_blocks_spec(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer) {
#if defined(USE_A3)
  LOG(FATAL) << "A3 does not support llmdatadist.";
  return false;
#else
  for (int64_t layer_index = 0; layer_index < spec_num_layers_; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);
    // Push the KV Cache computed at this layer for all requests to the
    // designated worker.
    for (const auto& pair : merged_kv_infos) {
      const KVCacheInfo& kv_info = pair.second;
      CacheIndex k_cache_index{kv_info.dst_cluster_id, spec_k_cache_.cache_id};
      CacheIndex v_cache_index{kv_info.dst_cluster_id, spec_v_cache_.cache_id};
      KvCacheExtParam ext_param{};
      ext_param.src_layer_range =
          std::pair<int32_t, int32_t>(layer_index, layer_index);
      ext_param.dst_layer_range =
          std::pair<int32_t, int32_t>(layer_index, layer_index);
      ext_param.tensor_num_per_layer = 1;

      CHECK_LDD_RET(llm_data_dist_->PushKvBlocks(spec_k_cache_,
                                                 k_cache_index,
                                                 kv_info.src_blocks,
                                                 kv_info.dst_blocks,
                                                 ext_param));
      CHECK_LDD_RET(llm_data_dist_->PushKvBlocks(spec_v_cache_,
                                                 v_cache_index,
                                                 kv_info.src_blocks,
                                                 kv_info.dst_blocks,
                                                 ext_param));
    }
  }
  return true;
#endif
}

bool SpecKVCacheTransfer::push_embed_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos) {
#if defined(USE_A3)
  LOG(FATAL) << "A3 does not support llmdatadist.";
  return false;
#else
  for (const auto& pair : merged_kv_infos) {
    const KVCacheInfo& kv_info = pair.second;
    CacheIndex cache_index{kv_info.dst_cluster_id, embed_cache_.cache_id};
    KvCacheExtParam ext_param{};
    ext_param.src_layer_range = std::pair<int32_t, int32_t>(0, 0);
    ext_param.dst_layer_range = std::pair<int32_t, int32_t>(0, 0);
    ext_param.tensor_num_per_layer = 1;
    CHECK_LDD_RET(llm_data_dist_->PushKvBlocks(embed_cache_,
                                               cache_index,
                                               kv_info.src_embed_ids,
                                               kv_info.dst_embed_ids,
                                               ext_param));
  }
  return true;
#endif
}
#endif

#if defined(USE_NPU)
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
#endif

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
      int64_t k_cache_id = info.remote_instance_info.k_cache_ids[i];
      int64_t v_cache_id = info.remote_instance_info.v_cache_ids[i];
      std::string key = std::to_string(dst_cluster_id) + "_" +
                        std::to_string(k_cache_id) + "_" +
                        std::to_string(v_cache_id);
      // Merge all kv blocks with the same destination worker into a single
      // vector.
      if (merged_kv_infos.find(key) == merged_kv_infos.end()) {
        KVCacheInfo kv_info;
        kv_info.dst_cluster_id = dst_cluster_id;
        kv_info.dst_k_cache_id = k_cache_id;
        kv_info.dst_v_cache_id = v_cache_id;
        kv_info.src_blocks.insert(kv_info.src_blocks.end(),
                                  info.local_blocks_ids.begin(),
                                  info.local_blocks_ids.end());
        kv_info.dst_blocks.insert(kv_info.dst_blocks.end(),
                                  info.remote_blocks_ids.begin(),
                                  info.remote_blocks_ids.end());
        kv_info.src_embed_ids.push_back(kv_info.src_blocks.back());
        kv_info.dst_embed_ids.push_back(kv_info.dst_blocks.back());
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
        merged_kv_infos[key].src_embed_ids.push_back(
            merged_kv_infos[key].src_blocks.back());
        merged_kv_infos[key].dst_embed_ids.push_back(
            merged_kv_infos[key].dst_blocks.back());
      }
    }
  }
}

void SpecKVCacheTransfer::copy_blocks(const std::vector<int>& blocks,
                                      bool h2d) {
  std::vector<uint64_t> _blocks;
  _blocks.reserve(blocks.size());
  for (const auto& block : blocks) {
    _blocks.push_back(static_cast<uint64_t>(block));
  }
  if (h2d) {
    CHECK_LDD_RET(llm_data_dist_->CopyKvBlocks(
        embed_host_cache_, embed_cache_, _blocks, {_blocks}));
  } else {
    CHECK_LDD_RET(llm_data_dist_->CopyKvBlocks(
        embed_cache_, embed_host_cache_, _blocks, {_blocks}));
  }
}
}  // namespace xllm
