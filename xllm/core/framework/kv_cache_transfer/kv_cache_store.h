#pragma once

#include <client.h>
#include <glog/logging.h>

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "util/slice.h"

namespace xllm {

class KVCache;

struct KVCacheStoreInitConfig {
  std::string protocol = "rdma";
  std::string localhost_name = "127.0.0.1";
  std::string metadata_server = "";
  std::string master_server_address = "";
  uint32_t tp_rank = 0;
  uint32_t tp_size = 0;
};

using StoreGroupedPrefixCaches =
    std::map<PrefixCacheGroup::Value, std::vector<const KVCache*>>;

class KVCacheStore {
 public:
  ~KVCacheStore();

  bool init(const KVCacheStoreInitConfig& config,
            const StoreGroupedPrefixCaches& kv_caches);

  uint32_t batch_put(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return batch_put(slice);
  }

  uint32_t batch_get(
      const std::vector<BlockTransferInfo>& block_transfer_info) {
    Slice<BlockTransferInfo> slice(block_transfer_info);
    return batch_get(slice);
  }

  uint32_t batch_put(Slice<BlockTransferInfo>& block_transfer_info);

  uint32_t batch_get(Slice<BlockTransferInfo>& block_transfer_info);

  static KVCacheStore& get_instance() {
    static KVCacheStore kvcache_store;
    return kvcache_store;
  }

 private:
  KVCacheStore() { rep_config_.replica_num = 1; }
  KVCacheStore(const KVCacheStore&) = delete;
  KVCacheStore& operator=(const KVCacheStore&) = delete;

  std::string genarate_mooncake_key(const std::string& hash_key,
                                    PrefixCacheGroup group,
                                    uint32_t tp_rank) const;
  std::vector<mooncake::Slice> genarate_mooncake_slice(uint32_t block_id,
                                                       PrefixCacheGroup group);
  bool register_memory(std::string location = "cpu:0");

  uint32_t batch_exist(std::vector<std::string>& keys);

 private:
  KVCacheStoreInitConfig config_;
  mooncake::ReplicateConfig rep_config_;

  StoreGroupedPrefixCaches kv_caches_;
  std::shared_ptr<mooncake::Client> client_ptr_;
};

}  // namespace xllm
