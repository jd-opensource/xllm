#pragma once

#include "embedding_allocator.h"
#include "llm_data_dist_transfer.h"

namespace xllm {

#if defined(USE_NPU)
using namespace llm_datadist;
#endif

class SpecKVCacheTransfer : public LlmDataDistTransfer {
 public:
  SpecKVCacheTransfer(const std::string& device_ip,
                      const uint16_t listen_port,
                      const InstanceRole& instance_role);

  virtual ~SpecKVCacheTransfer() = default;

  void allocate_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      const torch::ScalarType dtype) override;

  void allocate_kv_cache_spec(
      std::vector<xllm::KVCache>& kv_caches,
      const int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      torch::ScalarType dtype) override;

  void _allocate_kv_cache(
      std::vector<xllm::KVCache>& kv_caches,
      const int64_t num_layers,
      const std::vector<std::vector<int64_t>>& kv_cache_shape,
      torch::ScalarType dtype,
      bool is_spec,
      Cache& k_cache,
      Cache& v_cache);

  void allocate_embedding(
      std::shared_ptr<EmbeddingAllocator> embedding_allocator,
      const std::vector<int64_t>& embedding_shape,
      torch::ScalarType dtype,
      torch::Device device);

  void free_kv_cache() override;

  bool pull_kv_blocks(const uint64_t src_cluster_id,
                      const std::string& src_addr,
                      const int64_t src_k_cache_id,
                      const int64_t src_v_cache_id,
                      const std::vector<uint64_t>& src_blocks,
                      const std::vector<uint64_t>& dst_blocks) override;

  folly::SemiFuture<bool> push_kv_blocks_async(
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args,
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer,
      bool is_spec_draft) override;

#if defined(USE_NPU)
  bool push_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer,
      bool is_spec_draft) override;

  bool push_kv_blocks_spec(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      std::shared_ptr<NPULayerSynchronizerImpl>& layer_synchronizer);

  bool push_embed_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos);
#endif

  void merge_kv_blocks(
      std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
      const std::vector<TransferKVInfo>& transfer_kv_infos,
      const ParallelArgs& parallel_args) override;

  void copy_blocks(const std::vector<int>& blocks, bool h2d);

 private:
  int64_t spec_num_layers_;

#if defined(USE_NPU)
  Cache spec_k_cache_;
  Cache spec_v_cache_;
  Cache embed_cache_;
  Cache embed_host_cache_;

  Cache host_cache;
  std::vector<std::vector<uint16_t>> buffers;
#endif
};

}  // namespace xllm
