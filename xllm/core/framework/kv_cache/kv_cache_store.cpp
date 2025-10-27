
#include "kv_cache_store.h"

#include <Mooncake/mooncake-store/include/utils.h>
#include <glog/logging.h>

#include <string>
#include <unordered_map>

#include "util/hash_util.h"

namespace xllm {

bool KVCacheStore::init(const StoreConfig& config,
                        std::vector<xllm::KVCache>* host_kv_caches) {
  CHECK(!is_initialized_) << "KVCacheStore is initialized.";
  config_ = config;
  host_kv_caches_ = host_kv_caches;
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
                                             config_.metadata_connstring,
                                             config_.protocol,
                                             device_names,
                                             config_.master_server_entry);

  rep_config_.replica_num = config_.replica_num;
  // rep_config_.preferred_segment = config_.localhost_name;

  if (!client_opt.has_value()) {
    LOG(FATAL) << "mooncake::Client::Create fail! Failed to create client with "
                  "host_name: "
               << config_.localhost_name;
  }
  client_ptr_ = client_opt.value();

  auto k_tensor_one_block = host_kv_caches_->at(0).get_k_cache();
  auto v_tensor_one_block = host_kv_caches_->at(0).get_v_cache();

  k_cache_size_per_block_ =
      k_tensor_one_block.numel() * k_tensor_one_block.element_size();
  v_cache_size_per_block_ =
      v_tensor_one_block.numel() * v_tensor_one_block.element_size();

  LOG(INFO) << "k_cache_size_per_block: " << k_cache_size_per_block_;
  LOG(INFO) << "v_cache_size_per_block: " << v_cache_size_per_block_;

  if (config_.protocol == "rdma") {
    for (int block = 0; block < host_kv_caches_->size(); block++) {
      void* key_cache = static_cast<char*>(
          host_kv_caches_->at(block).get_k_cache().data_ptr());

      auto register_k_result = client_ptr_->RegisterLocalMemory(
          key_cache, k_cache_size_per_block_, "cpu:0", false, false);

      if (!register_k_result.has_value()) {
        LOG(ERROR) << "Failed to register local memory for key cache: "
                   << toString(register_k_result.error());
        return false;
      }

      void* value_cache = static_cast<char*>(
          host_kv_caches_->at(block).get_v_cache().data_ptr());

      auto register_v_result = client_ptr_->RegisterLocalMemory(
          value_cache, v_cache_size_per_block_, "cpu:0", false, false);

      if (!register_v_result.has_value()) {
        LOG(ERROR) << "Failed to register local memory for value cache: "
                   << toString(register_v_result.error());
        return false;
      }
    }
  }
  is_initialized_ = true;
  return true;
}

KVCacheStore::~KVCacheStore() {
  if (client_ptr_) {
    client_ptr_.reset();
  }
}

uint32_t KVCacheStore::batch_put(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (!is_initialized_) {
    return 0;
  }
  std::vector<std::string> str_keys;
  std::vector<std::vector<mooncake::Slice>> slices;

  str_keys.reserve(block_transfer_info.size());
  slices.reserve(block_transfer_info.size());
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);

    str_key.append(std::to_string(config_.tp_rank));

    auto exist = client_ptr_->IsExist(str_key);
    if (exist.has_value() && exist.value()) {
      continue;
    }

    str_keys.emplace_back(str_key);

    void* k_cache =
        host_kv_caches_->at(block_info.dst_block_id).get_k_cache().data_ptr();
    void* v_cache =
        host_kv_caches_->at(block_info.dst_block_id).get_k_cache().data_ptr();

    slices.emplace_back(std::vector<mooncake::Slice>{
        mooncake::Slice{k_cache, k_cache_size_per_block_},
        mooncake::Slice{v_cache, v_cache_size_per_block_}});
  }

  if (str_keys.size() == 0) {
    return block_transfer_info.size();
  }

  uint64_t success_cnt = block_transfer_info.size() - str_keys.size();
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
  if (!is_initialized_) {
    return 0;
  }
  std::unordered_map<std::string, std::vector<mooncake::Slice>> slices;
  std::vector<std::string> str_keys;

  str_keys.reserve(block_transfer_info.size());
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);

    str_key.append(std::to_string(config_.tp_rank));
    auto exist = client_ptr_->IsExist(str_key);
    if (!exist.has_value() || !exist.value()) {
      break;
    }

    str_keys.emplace_back(str_key);

    void* k_cache =
        host_kv_caches_->at(block_info.dst_block_id).get_k_cache().data_ptr();
    void* v_cache =
        host_kv_caches_->at(block_info.dst_block_id).get_k_cache().data_ptr();

    slices.insert(
        std::make_pair(str_key,
                       std::vector<mooncake::Slice>{
                           mooncake::Slice{k_cache, k_cache_size_per_block_},
                           mooncake::Slice{v_cache, v_cache_size_per_block_}}));
  }

  if (str_keys.size() == 0) {
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

uint32_t KVCacheStore::batch_remove(
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(is_initialized_) << "KVCacheStore is not initialized.";
  uint32_t success_cnt = 0;
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);
    str_key.append(std::to_string(config_.tp_rank));

    auto result = client_ptr_->Remove(str_key);

    if (result.has_value()) {
      success_cnt++;
    }
  }
  return success_cnt;
}

uint32_t KVCacheStore::batch_exist(std::vector<std::string>&& keys) {
  if (!is_initialized_) {
    return 0;
  }
  auto exist_vec = client_ptr_->BatchIsExist(std::move(keys));
  uint32_t ret = 0;
  for (auto exist : exist_vec) {
    if (!exist.has_value() || !exist.value()) {
      break;
    }
    ret++;
  }
  return ret;
}

}  // namespace xllm
