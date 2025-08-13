
#include "kv_cache_store.h"

#include <MoonCake/mooncake-store/include/utils.h>
#include <glog/logging.h>

#include <string>
#include <unordered_map>

#include "util/hash_util.h"

namespace llm {

KVCacheStore::KVCacheStore(const StoreConfig& config,
                           std::vector<llm::KVCache>* host_kv_caches)
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
  rep_config_.preferred_segment = config_.localhost_name;

  if (!client_opt.has_value()) {
    LOG(ERROR) << "mooncake::Client::Create fail!";
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

uint64_t KVCacheStore::batch_put(const std::vector<CacheContent>& blocks) {
  std::vector<std::string> str_keys;
  std::vector<std::vector<mooncake::Slice>> slices;

  str_keys.reserve(blocks.size());
  slices.resize(blocks.size());
  for (auto block : blocks) {
    std::string str_key(reinterpret_cast<const char*>(block.hash_key),
                        MURMUR_HASH3_VALUE_LEN);
    str_key.append(std::to_string(config_.tp_rank));

    str_keys.emplace_back(str_key);

    std::vector<mooncake::Slice> slice;
    slice.reserve(host_kv_caches_->size() * 2);
    for (int layer = 0; layer < host_kv_caches_->size(); layer++) {
      void* key_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_k_cache().data_ptr()) +
          block.host_block_id * key_cache_size_per_layer_;
      slice.emplace_back(mooncake::Slice{key_cache, key_cache_size_per_layer_});

      void* value_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_v_cache().data_ptr()) +
          block.host_block_id * value_cache_size_per_layer_;
      slice.emplace_back(
          mooncake::Slice{value_cache, value_cache_size_per_layer_});
      slices.emplace_back(std::move(slice));
    }
  }

  auto results = client_ptr_->BatchPut(str_keys, slices, rep_config_);
  uint64_t success_cnt = 0;
  for (int i = 0; i < blocks.size(); i++) {
    if (!results[i].has_value()) {
      success_cnt = i;
      break;
    }
  }
  return success_cnt;
}

uint64_t KVCacheStore::batch_get(const std::vector<CacheContent>& blocks) {
  std::unordered_map<std::string, std::vector<mooncake::Slice>> slices;
  std::vector<std::string> str_keys;

  str_keys.reserve(blocks.size());
  for (auto block : blocks) {
    std::string str_key(reinterpret_cast<const char*>(block.hash_key),
                        MURMUR_HASH3_VALUE_LEN);
    str_key.append(std::to_string(config_.tp_rank));

    str_keys.emplace_back(str_key);

    slices.insert(std::make_pair(str_key, std::vector<mooncake::Slice>()));

    slices[str_key].reserve(host_kv_caches_->size() * 2);
    for (int layer = 0; layer < host_kv_caches_->size(); layer++) {
      void* key_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_k_cache().data_ptr()) +
          block.host_block_id * key_cache_size_per_layer_;
      slices[str_key].emplace_back(
          mooncake::Slice{key_cache, key_cache_size_per_layer_});

      void* value_cache =
          static_cast<char*>(
              host_kv_caches_->at(layer).get_v_cache().data_ptr()) +
          block.host_block_id * value_cache_size_per_layer_;
      slices[str_key].emplace_back(
          mooncake::Slice{value_cache, value_cache_size_per_layer_});
    }
  }

  auto results = client_ptr_->BatchGet(str_keys, slices);
  uint64_t success_cnt = 0;
  for (int i = 0; i < blocks.size(); i++) {
    if (!results[i].has_value()) {
      success_cnt = i;
      break;
    }
  }
  return success_cnt;
}

uint64_t KVCacheStore::batch_remove(const std::vector<CacheContent>& blocks) {
  uint64_t success_cnt = 0;
  for (auto block : blocks) {
    std::string str_key(reinterpret_cast<const char*>(block.hash_key),
                        MURMUR_HASH3_VALUE_LEN);
    str_key.append(std::to_string(config_.tp_rank));

    auto result = client_ptr_->Remove(str_key);

    if (result.has_value()) {
      success_cnt++;
    }
  }
  return success_cnt;
}

}  // namespace llm
