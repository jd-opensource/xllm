#pragma once

#include <Mooncake/mooncake-store/include/client.h>
#include <glog/logging.h>

#include <string>

#include "common/macros.h"
#include "framework/model/model_input_params.h"
#include "kv_cache.h"
#include "util/slice.h"

namespace xllm {

struct StoreConfig {
  std::string localhost_name = "127.0.0.1";
  std::string protocol = "tcp";
  std::string metadata_server = "";
  std::string master_server_address = "";
  int replica_num = 1;
  uint32_t tp_rank = 0;
};

class KVCacheStore {
 public:
  ~KVCacheStore();

  bool init(const StoreConfig& config,
            std::vector<xllm::KVCache>* host_kv_caches);

  uint32_t batch_put(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    return batch_put({block_transfer_info});
  }

  uint32_t batch_get(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    return batch_get({block_transfer_info});
  }

  uint32_t batch_remove(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    return batch_remove({block_transfer_info});
  }

  uint32_t batch_put(Slice<BlockTransferInfo>& block_transfer_info);

  uint32_t batch_get(Slice<BlockTransferInfo>& block_transfer_info);

  uint32_t batch_remove(Slice<BlockTransferInfo>& block_transfer_info);

  uint32_t batch_exist(std::vector<std::string>&& keys);

  static KVCacheStore& get_instance() {
    static KVCacheStore kvcache_store;
    return kvcache_store;
  }

 private:
  KVCacheStore() = default;
  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

 private:
  bool is_initialized_ = false;

  StoreConfig config_;
  mooncake::ReplicateConfig rep_config_;

  std::vector<xllm::KVCache>* host_kv_caches_;

  uint64_t k_cache_size_per_block_;
  uint64_t v_cache_size_per_block_;

  std::shared_ptr<mooncake::Client> client_ptr_;
};

}  // namespace xllm
