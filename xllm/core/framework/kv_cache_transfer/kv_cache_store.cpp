#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <Mooncake/mooncake-store/include/utils.h>
#if defined(USE_NPU)
#include <Mooncake/mooncake-transfer-engine/include/transport/ub_transport/memfabric_smem_dl.h>
#endif
#include <glog/logging.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>

#include "util/hash_util.h"

namespace xllm {

bool KVCacheStore::init(const KVCacheStoreInitConfig& config,
                        const GroupedKVCaches& kv_caches) {
  CHECK(!client_ptr_) << "KVCacheStore is initialized.";
  config_ = config;
  kv_caches_.clear();
  for (const auto& group_caches : kv_caches) {
    CHECK(group_caches.first != KVCacheTensorGroup::INVALID)
        << "KVCacheStore does not accept INVALID group.";
    kv_caches_[static_cast<KVCacheTensorGroup::Value>(group_caches.first)] =
        group_caches.second;
  }

  std::optional<std::string> device_names = std::nullopt;
  if (config_.protocol == "rdma") {
    if (getenv("DEVICE_NAMES")) {
      device_names = getenv("DEVICE_NAMES");
      LOG(INFO) << "device_names: " << device_names.value();
    } else {
      LOG(WARNING) << "env DEVICE_NAME not exist, set protocol as tcp";
      config_.protocol = "tcp";
    }
  }
#if defined(USE_NPU)
  mooncake::MemFabricSmemDl::SetSmemTypeFlag(mooncake::SMEM_BM);
#endif

  auto client_opt = mooncake::Client::Create(config_.localhost_name,
                                             config_.metadata_server,
                                             config_.protocol,
                                             device_names,
                                             config_.master_server_address);
  CHECK(!client_opt.has_value())
      << "Failed to create mooncake client with host_name: "
      << config_.localhost_name;

  client_ptr_ = client_opt.value();

  if (config_.protocol == "rdma") {
    CHECK(config_.page_aligned_tensor_data != nullptr)
        << "KVCacheStore rdma RegisterLocalMemory got invalide data pointer: "
        << config_.page_aligned_tensor_data;
    CHECK(config_.page_aligned_total_size > 0)
        << "KVCacheStore rdma RegisterLocalMemory got invalide data size: "
        << config_.page_aligned_total_size;
    CHECK(register_memory(config_.page_aligned_tensor_data,
                          config_.page_aligned_total_size,
                          "cpu:0"));
  }

  return true;
}

KVCacheStore::~KVCacheStore() {
  if (client_ptr_) {
    client_ptr_.reset();
  }
}

uint32_t KVCacheStore::batch_put(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!client_ptr_) {
    return 0;
  }
  std::vector<std::string> str_keys;
  std::vector<std::vector<mooncake::Slice>> slices;
  uint32_t success_cnt = 0;

  str_keys.reserve(block_transfer_info.size());
  slices.reserve(block_transfer_info.size());
  for (size_t info_index = 0; info_index < block_transfer_info.size();
       info_index++) {
    const auto& block_info = block_transfer_info[info_index];
    str_keys.emplace_back(genarate_mooncake_key(
        std::string(reinterpret_cast<const char*>(block_info.hash_key),
                    XXH3_128BITS_HASH_VALUE_LEN),
        block_info.group,
        config_.tp_rank));
    slices.emplace_back(
        genarate_mooncake_slice(block_info.src_block_id, block_info.group));
  }

  if (str_keys.empty()) {
    return success_cnt;
  }

  auto results = client_ptr_->BatchPut(str_keys, slices, rep_config_);

  for (int i = 0; i < str_keys.size(); i++) {
    if (!results[i].has_value()) {
      break;
    }
    success_cnt++;
  }
  return success_cnt;
}

uint32_t KVCacheStore::batch_get(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!client_ptr_) {
    return 0;
  }
  std::unordered_map<std::string, std::vector<mooncake::Slice>> slices;
  std::vector<std::string> str_keys;

  str_keys.reserve(block_transfer_info.size());
  for (const auto& block_info : block_transfer_info) {
    std::string str_key = genarate_mooncake_key(
        std::string(reinterpret_cast<const char*>(block_info.hash_key),
                    XXH3_128BITS_HASH_VALUE_LEN),
        block_info.group,
        config_.tp_rank);
    str_keys.emplace_back(str_key);
    slices.insert(std::make_pair(
        str_key,
        genarate_mooncake_slice(block_info.dst_block_id, block_info.group)));
  }

  if (str_keys.empty()) {
    return 0;
  }

  uint64_t success_cnt = 0;
  auto results = client_ptr_->BatchGet(str_keys, slices);
  for (int i = 0; i < str_keys.size(); i++) {
    if (!results[i].has_value()) {
      break;
    }
    success_cnt++;
  }
  return success_cnt;
}

std::string KVCacheStore::genarate_mooncake_key(const std::string& hash_key,
                                                KVCacheTensorGroup group,
                                                uint32_t tp_rank) const {
  std::string str_key(hash_key);
  str_key.append("-");
  str_key.append(std::to_string(tp_rank));
  str_key.append("-");
  str_key.append(group.to_string());
  return str_key;
}

std::vector<mooncake::Slice> KVCacheStore::genarate_mooncake_slice(
    uint32_t block_id,
    KVCacheTensorGroup group) {
  std::vector<mooncake::Slice> slice;
  const auto group_it =
      kv_caches_.find(static_cast<KVCacheTensorGroup::Value>(group));

  const auto& kv_caches = group_it->second;
  switch (config_.format) {
    case TensorFormat::BLOCK_WISE: {
      CHECK_LT(static_cast<size_t>(block_id), kv_caches.size())
          << "Block id out of range: " << block_id;

      for (const auto& cache_tensor :
           kv_caches[block_id]->get_cache_tensors()) {
        if (group.contains(cache_tensor.role) &&
            cache_tensor.tensor.defined()) {
          slice.emplace_back(
              mooncake::Slice{cache_tensor.tensor.data_ptr(),
                              static_cast<size_t>(cache_tensor.tensor.numel()) *
                                  cache_tensor.tensor.element_size()});
        }
      }
      return slice;
    }
    case TensorFormat::LAYER_WISE: {
      for (size_t i = 0; i < kv_caches.size(); i++) {
        CHECK(kv_caches[i] != nullptr)
            << "Null KV cache in group: " << group.to_string()
            << ", layer_id: " << i;

        for (const auto& cache_tensor : kv_caches[i]->get_cache_tensors()) {
          if (group.contains(cache_tensor.role) &&
              cache_tensor.tensor.defined()) {
            slice.emplace_back(mooncake::Slice{
                cache_tensor.tensor[block_id].data_ptr(),
                static_cast<size_t>(cache_tensor.tensor[block_id].numel()) *
                    cache_tensor.tensor[block_id].element_size()});
          }
        }
      }
      return slice;
    }
    default:
      LOG(FATAL) << "Unrecognized tensor format!";
      break;
  }
}

bool KVCacheStore::register_memory(void* data,
                                   size_t length,
                                   std::string location) {
  auto result =
      client_ptr_->RegisterLocalMemory(data, length, location, false, false);
  if (!result.has_value()) {
    LOG(ERROR) << "Failed to register local memory: "
               << toString(result.error());
    return false;
  }
  return true;
}

}  // namespace xllm
