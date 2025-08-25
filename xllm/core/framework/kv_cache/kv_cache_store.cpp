
#include "kv_cache_store.h"

#include <Mooncake/mooncake-store/include/utils.h>
#include <glog/logging.h>

#include <string>
#include <unordered_map>

#include "util/hash_util.h"

namespace xllm {

KVCacheStore::KVCacheStore(const StoreConfig& config,
                           std::vector<xllm::KVCache>* host_kv_caches)
    : config_(config), host_kv_caches_(host_kv_caches) {
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
    return;
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

  for (int layer = 0; layer < host_kv_caches_->size(); layer++) {
    void* key_cache =
        static_cast<char*>(host_kv_caches_->at(layer).get_k_cache().data_ptr());

    auto register_k_result = client_ptr_->RegisterLocalMemory(
        key_cache, key_cache_host_size, "cpu:0", false, false);

    if (!register_k_result.has_value()) {
      LOG(ERROR) << "Failed to register local memory for key cache: "
                 << toString(register_k_result.error());
      return;
    }

    void* value_cache =
        static_cast<char*>(host_kv_caches_->at(layer).get_v_cache().data_ptr());

    auto register_v_result = client_ptr_->RegisterLocalMemory(
        value_cache, value_cache_host_size, "cpu:0", false, false);

    if (!register_v_result.has_value()) {
      LOG(ERROR) << "Failed to register local memory for value cache: "
                 << toString(register_v_result.error());
      return;
    }
  }
}

KVCacheStore::~KVCacheStore() {
  if (client_ptr_) {
    client_ptr_.reset();
  }
}

uint64_t KVCacheStore::batch_put(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  std::vector<std::string> str_keys;
  std::vector<std::vector<mooncake::Slice>> slices;

  str_keys.reserve(cache_block_info.size());
  slices.reserve(cache_block_info.size());
  for (auto cache_content : cache_block_info) {
    std::string str_key(reinterpret_cast<const char*>(cache_content.hash_key),
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
          cache_content.host_block_id * key_cache_size_per_layer_;
      slice.emplace_back(mooncake::Slice{key_cache, key_cache_size_per_layer_});

      void* value_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_v_cache().data_ptr()) +
          cache_content.host_block_id * value_cache_size_per_layer_;
      slice.emplace_back(
          mooncake::Slice{value_cache, value_cache_size_per_layer_});
    }
    slices.emplace_back(std::move(slice));
  }

  if (str_keys.size() == 0) {
    return cache_block_info.size();
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

uint64_t KVCacheStore::batch_get(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  std::unordered_map<std::string, std::vector<mooncake::Slice>> slices;
  std::vector<std::string> str_keys;

  str_keys.reserve(cache_block_info.size());
  for (auto cache_content : cache_block_info) {
    std::string str_key(reinterpret_cast<const char*>(cache_content.hash_key),
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
          cache_content.host_block_id * key_cache_size_per_layer_;
      slices[str_key].emplace_back(
          mooncake::Slice{key_cache, key_cache_size_per_layer_});

      void* value_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_v_cache().data_ptr()) +
          cache_content.host_block_id * value_cache_size_per_layer_;
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

uint64_t KVCacheStore::batch_remove(
    const std::vector<CacheBlockInfo>& cache_block_info) {
  uint64_t success_cnt = 0;
  for (auto cache_content : cache_block_info) {
    std::string str_key(reinterpret_cast<const char*>(cache_content.hash_key),
                        MURMUR_HASH3_VALUE_LEN);
    str_key.append(std::to_string(config_.tp_rank));

    auto result = client_ptr_->Remove(str_key);

    if (result.has_value()) {
      success_cnt++;
    }
  }
  return success_cnt;
}

}  // namespace xllm
