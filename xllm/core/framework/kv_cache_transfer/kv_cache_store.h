#pragma once

#include <Mooncake/mooncake-store/include/client_service.h>
#include <glog/logging.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_tensor_group.h"
#include "framework/model/model_input_params.h"
#include "util/slice.h"

namespace xllm {

enum class TensorFormat : uint8_t {
  LAYER_WISE =
      0,  // tensor shape: layer_num * [block_num, block_size, header, dim]
  BLOCK_WISE =
      1  // tensor shape: block_num * [layer_num, block_size, header, dim]
};

struct KVCacheStoreInitConfig {
  std::string protocol = "rdma";
  std::string localhost_name = "127.0.0.1";
  std::string metadata_server = "";
  std::string master_server_address = "";
  uint32_t tp_rank = 0;
  uint32_t tp_size = 0;
  size_t page_aligned_total_size = 0;
  void* page_aligned_tensor_data = nullptr;
  TensorFormat format = TensorFormat::BLOCK_WISE;
};

using GroupedKVCaches =
    std::vector<std::pair<KVCacheTensorGroup, std::vector<xllm::KVCache*>>>;
using GroupedKVCacheHashKeys =
    std::map<KVCacheTensorGroup, std::vector<std::string>>;

class KVCacheStore {
 public:
  ~KVCacheStore();

  bool init(const KVCacheStoreInitConfig& config,
            const GroupedKVCaches& kv_caches);

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

  using GroupedKVCacheMap =
      std::map<KVCacheTensorGroup::Value, std::vector<xllm::KVCache*>>;

  std::string genarate_mooncake_key(const std::string& hash_key,
                                    KVCacheTensorGroup group,
                                    uint32_t tp_rank) const;
  std::vector<mooncake::Slice> genarate_mooncake_slice(
      uint32_t block_id,
      KVCacheTensorGroup group);
  bool register_memory(void* data, size_t length, std::string location = "");

 private:
  bool is_initialized_ = false;

  KVCacheStoreInitConfig config_;
  mooncake::ReplicateConfig rep_config_;

  GroupedKVCacheMap kv_caches_;
  std::shared_ptr<mooncake::Client> client_ptr_;
};

}  // namespace xllm
