#include "framework/kv_cache_transfer/kv_cache_store.h"

#include <glog/logging.h>
#include <utils.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>

#include "util/hash_util.h"

namespace xllm {
namespace {

uint32_t resolve_put_block_id(const BlockTransferInfo& block_info) {
  switch (block_info.transfer_type) {
    case TransferType::D2H2G:
      return static_cast<uint32_t>(block_info.dst_block_id);
    case TransferType::D2G:
      return static_cast<uint32_t>(block_info.src_block_id);
    default:
      LOG(FATAL) << "Unsupported transfer type for batch_put: "
                 << static_cast<uint32_t>(block_info.transfer_type);
      return 0;
  }
}

uint32_t resolve_get_block_id(const BlockTransferInfo& block_info) {
  switch (block_info.transfer_type) {
    case TransferType::G2H:
      return static_cast<uint32_t>(block_info.dst_block_id);
    case TransferType::G2D:
      return static_cast<uint32_t>(block_info.dst_block_id);
    default:
      LOG(FATAL) << "Unsupported transfer type for batch_get: "
                 << static_cast<uint32_t>(block_info.transfer_type);
      return 0;
  }
}

}  // namespace

bool KVCacheStore::init(const KVCacheStoreInitConfig& config,
                        const StoreGroupedPrefixCaches& kv_caches) {
  CHECK(!client_ptr_) << "KVCacheStore is initialized.";
  config_ = config;
  kv_caches_ = kv_caches;
  CHECK(!kv_caches_.empty()) << "KVCacheStore requires non-empty kv caches.";

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
  auto client_opt = mooncake::Client::Create(config_.localhost_name,
                                             config_.metadata_server,
                                             config_.protocol,
                                             device_names,
                                             config_.master_server_address);
  CHECK(client_opt.has_value())
      << "Failed to create mooncake client with host_name: "
      << config_.localhost_name;

  client_ptr_ = client_opt.value();

  if (config_.protocol == "rdma") {
    CHECK(register_memory("cpu:0"));
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
    const std::string hash_key(
        reinterpret_cast<const char*>(block_info.hash_key),
        XXH3_128BITS_HASH_VALUE_LEN);
    const PrefixCacheGroup group = block_info.group;
    const uint32_t cache_block_id = resolve_put_block_id(block_info);
    str_keys.emplace_back(
        genarate_mooncake_key(hash_key, group, config_.tp_rank));
    slices.emplace_back(genarate_mooncake_slice(cache_block_id, group));
  }

  if (str_keys.empty()) {
    return success_cnt;
  }

  auto results = client_ptr_->BatchPut(str_keys, slices, rep_config_);
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].has_value()) {
      continue;
    }
    ++success_cnt;
  }
  return success_cnt;
}

uint32_t KVCacheStore::batch_get(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!client_ptr_) {
    return 0;
  }
  std::unordered_map<std::string, std::vector<mooncake::Slice>> slices_by_key;
  std::vector<std::string> str_keys;
  bool has_single_group = false;

  str_keys.reserve(block_transfer_info.size());
  for (size_t info_index = 0; info_index < block_transfer_info.size();
       ++info_index) {
    const auto& block_info = block_transfer_info[info_index];
    const std::string hash_key(
        reinterpret_cast<const char*>(block_info.hash_key),
        XXH3_128BITS_HASH_VALUE_LEN);
    const PrefixCacheGroup group = block_info.group;
    const uint32_t cache_block_id = resolve_get_block_id(block_info);
    std::string str_key =
        genarate_mooncake_key(hash_key, group, config_.tp_rank);
    if (group == PrefixCacheGroup::SINGLE) {
      has_single_group = true;
    }
    str_keys.emplace_back(str_key);
    slices_by_key.insert(std::make_pair(
        str_key, genarate_mooncake_slice(cache_block_id, group)));
  }

  if (str_keys.empty()) {
    return 0;
  }

  if (has_single_group && batch_exist(str_keys) != str_keys.size()) {
    return 0;
  }

  auto results = client_ptr_->BatchGet(str_keys, slices_by_key);
  uint32_t success_cnt = 0;
  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].has_value()) {
      if (has_single_group) {
        return 0;
      }
      continue;
    }
    ++success_cnt;
  }
  return success_cnt;
}

uint32_t KVCacheStore::batch_exist(std::vector<std::string>& keys) {
  if (!client_ptr_) {
    return 0;
  }
  auto exist_vec = client_ptr_->BatchIsExist(keys);
  uint32_t ret = 0;
  for (auto exist : exist_vec) {
    if (!exist.has_value() || !exist.value()) {
      break;
    }
    ret++;
  }
  return ret;
}

std::string KVCacheStore::genarate_mooncake_key(const std::string& hash_key,
                                                PrefixCacheGroup group,
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
    PrefixCacheGroup group) {
  std::vector<mooncake::Slice> slice;
  const auto& kv_caches =
      kv_caches_[static_cast<PrefixCacheGroup::Value>(group)];
  CHECK(!kv_caches.empty()) << "prefix cache group is empty.";
  for (size_t cache_idx = 0; cache_idx < kv_caches.size(); ++cache_idx) {
    const KVCache* kv_cache = kv_caches[cache_idx];
    CHECK(kv_cache != nullptr) << "prefix cache must not be null.";
    const PrefixCacheTensorMap tensor_map =
        kv_cache->get_prefix_cache_tensors(group, block_id);
    CHECK(!tensor_map.empty())
        << "prefix tensor map is empty for group " << group.to_string()
        << ", block_id=" << block_id << ", cache_idx=" << cache_idx;
    for (const auto& item : tensor_map) {
      const torch::Tensor& tensor = item.second;
      if (tensor.defined()) {
        slice.emplace_back(mooncake::Slice{
            tensor.data_ptr(),
            static_cast<size_t>(tensor.numel()) * tensor.element_size()});
      }
    }
  }
  return slice;
}

bool KVCacheStore::register_memory(std::string location) {
  CHECK(client_ptr_) << "KVCacheStore client is not initialized.";
  for (const auto& item : kv_caches_) {
    for (const KVCache* kv_cache : item.second) {
      CHECK(kv_cache != nullptr) << "prefix cache must not be null.";
      for (const HostPageAlignedRegion& region :
           kv_cache->get_host_page_aligned_regions()) {
        CHECK(region.base_ptr != nullptr)
            << "KVCacheStore rdma RegisterLocalMemory got invalid region ptr.";
        CHECK_GT(region.total_bytes, static_cast<size_t>(0))
            << "KVCacheStore rdma RegisterLocalMemory got invalid region size.";
        auto result = client_ptr_->RegisterLocalMemory(
            region.base_ptr, region.total_bytes, location, false, false);
        if (!result.has_value()) {
          LOG(ERROR) << "Failed to register local memory: "
                     << toString(result.error());
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace xllm
