#pragma once

#include <Mooncake/mooncake-store/include/client.h>
#include <glog/logging.h>

#include <string>

#include "common/macros.h"
#include "framework/model/model_input_params.h"
#include "kv_cache.h"
#include "util/slice.h"

namespace xllm {

enum class TensorFormat : uint8_t {
  LAYER_WISE =
      0,  // tensor shape: layer_num * [block_num, block_size, header, dim]
  BLOCK_WISE =
      1  // tensor shape: block_num * [layer_num, block_size, header, dim]
};

struct StoreConfig {
  std::string localhost_name = "127.0.0.1";
  std::string protocol = "tcp";
  std::string metadata_server = "";
  std::string master_server_address = "";
  int replica_num = 1;
  bool enable_mla = false;
  uint32_t tp_rank = 0;
  uint32_t tp_size = 0;
  uint32_t layers_wise_copy_batchs = 1;
  size_t total_size = 0;
  void* tensor_data = nullptr;
  TensorFormat format = TensorFormat::BLOCK_WISE;
};

class KVCacheStore {
 public:
  ~KVCacheStore();

  bool init(const StoreConfig& config, std::vector<xllm::KVCache>* kv_caches);

  uint32_t batch_put(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return batch_put(slice);
  }

  uint32_t batch_get(const std::vector<BlockTransferInfo>& block_transfer_info,
                     uint32_t slice_idx = 0) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return batch_get(slice, slice_idx);
  }

  uint32_t batch_put(Slice<BlockTransferInfo>& block_transfer_info);

  uint32_t batch_get(Slice<BlockTransferInfo>& block_transfer_info,
                     uint32_t slice_idx = 0);

  uint32_t batch_exist(std::vector<std::string>& keys);

  static KVCacheStore& get_instance() {
    static KVCacheStore kvcache_store;
    return kvcache_store;
  }

 private:
  KVCacheStore() = default;
  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  std::vector<mooncake::Slice> genarate_mooncake_slice(uint32_t block_id,
                                                       uint32_t slice_idx);
  bool register_memory(void* data, size_t length, std::string location = "");

 private:
  bool is_initialized_ = false;

  StoreConfig config_;
  mooncake::ReplicateConfig rep_config_;

  std::vector<xllm::KVCache>* kv_caches_;

  std::vector<std::pair<uint32_t, uint32_t>> layer_copy_index_pairs_;

  uint64_t k_cache_size_per_slice_ = 0;
  uint64_t v_cache_size_per_slice_ = 0;
  uint64_t index_cache_size_per_slice_ = 0;

  std::shared_ptr<mooncake::Client> client_ptr_;
};

}  // namespace xllm
