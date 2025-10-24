
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
  if (config_.protocol == "rdma") {
    if (getenv("DEVICE_NAME")) {
      auto name = getenv("DEVICE_NAME");
      LOG(INFO) << "device name: " << name;
      args_ = mooncake::rdma_args(name);
    } else {
      LOG(WARNING) << "env DEVICE_NAME not exist, set protocol as tcp";
      config_.protocol = "tcp";
      args_ = nullptr;
    }
  }

  auto client_opt = mooncake::Client::Create(config_.localhost_name,
                                             config_.metadata_connstring,
                                             config_.protocol,
                                             args_,
                                             config_.master_server_entry);

  rep_config_.replica_num = config_.replica_num;
  // rep_config_.preferred_segment = config_.localhost_name;

  if (!client_opt.has_value()) {
    LOG(FATAL) << "mooncake::Client::Create fail!";
    return false;
  }
  client_ptr_ = client_opt.value();

  auto key_tensor_one_layer = host_kv_caches_->at(0).get_k_cache();
  auto value_tensor_one_layer = host_kv_caches_->at(0).get_v_cache();

  key_cache_size_per_layer_ =
      key_tensor_one_layer[0].numel() * key_tensor_one_layer[0].element_size();
  value_cache_size_per_layer_ = value_tensor_one_layer[0].numel() *
                                value_tensor_one_layer[0].element_size();

  auto key_cache_host_size =
      key_tensor_one_layer.numel() * key_tensor_one_layer.element_size();
  auto value_cache_host_size =
      value_tensor_one_layer.numel() * value_tensor_one_layer.element_size();

  LOG(INFO) << "key_cache_size_per_layer: " << key_cache_size_per_layer_;
  LOG(INFO) << "value_cache_size_per_layer: " << value_cache_size_per_layer_;

  if (config_.protocol == "rdma") {
    for (int layer = 0; layer < host_kv_caches_->size(); layer++) {
      void* key_cache = static_cast<char*>(
          host_kv_caches_->at(layer).get_k_cache().data_ptr());

      auto register_k_result = client_ptr_->RegisterLocalMemory(
          key_cache, key_cache_host_size, "cpu:0", false, false);

      if (!register_k_result.has_value()) {
        LOG(ERROR) << "Failed to register local memory for key cache: "
                   << toString(register_k_result.error());
        return false;
      }

      void* value_cache = static_cast<char*>(
          host_kv_caches_->at(layer).get_v_cache().data_ptr());

      auto register_v_result = client_ptr_->RegisterLocalMemory(
          value_cache, value_cache_host_size, "cpu:0", false, false);

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

    std::vector<mooncake::Slice> slice;
    slice.reserve(host_kv_caches_->size() * 2);
    for (int layer = 0; layer < host_kv_caches_->size(); layer++) {
      void* key_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_k_cache().data_ptr()) +
          block_info.dst_block_id * key_cache_size_per_layer_;
      slice.emplace_back(mooncake::Slice{key_cache, key_cache_size_per_layer_});

      void* value_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_v_cache().data_ptr()) +
          block_info.dst_block_id * value_cache_size_per_layer_;
      slice.emplace_back(
          mooncake::Slice{value_cache, value_cache_size_per_layer_});
    }
    slices.emplace_back(std::move(slice));
  }

  if (str_keys.size() == 0) {
    return block_transfer_info.size();
  }

  uint64_t success_cnt = str_keys.size();
  auto results = client_ptr_->BatchPut(str_keys, slices, rep_config_);

  for (int i = 0; i < str_keys.size(); i++) {
    if (!results[i].has_value()) {
      success_cnt = i;
      // LOG(ERROR) << "success_cnt: " << success_cnt
      //            << ", failed to BatchPut: " << toString(results[i].error());
      break;
    }
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

    slices.insert(std::make_pair(str_key, std::vector<mooncake::Slice>()));

    slices[str_key].reserve(host_kv_caches_->size() * 2);
    for (int layer = 0; layer < host_kv_caches_->size(); layer++) {
      void* key_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_k_cache().data_ptr()) +
          block_info.dst_block_id * key_cache_size_per_layer_;
      slices[str_key].emplace_back(
          mooncake::Slice{key_cache, key_cache_size_per_layer_});

      void* value_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_v_cache().data_ptr()) +
          block_info.dst_block_id * value_cache_size_per_layer_;
      slices[str_key].emplace_back(
          mooncake::Slice{value_cache, value_cache_size_per_layer_});
    }
  }

  if (str_keys.size() == 0) {
    return 0;
  }

  uint64_t success_cnt = str_keys.size();
  auto results = client_ptr_->BatchGet(str_keys, slices);
  for (int i = 0; i < str_keys.size(); i++) {
    if (!results[i].has_value()) {
      success_cnt = i;
      // LOG(ERROR) << "success_cnt: " << success_cnt
      //            << ", failed to BatchGet: " << toString(results[i].error());
      break;
    }
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
