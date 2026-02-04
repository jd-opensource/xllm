
#include "kv_cache_store.h"

#include <Mooncake/mooncake-store/include/utils.h>
#if defined(USE_NPU)
#include <Mooncake/mooncake-transfer-engine/include/transport/ub_transport/memfabric_smem_dl.h>
#endif
#include <glog/logging.h>

#include <string>
#include <unordered_map>

#include "util/hash_util.h"

namespace xllm {

bool KVCacheStore::init(const StoreConfig& config,
                        std::vector<xllm::KVCache>* kv_caches) {
  CHECK(!is_initialized_) << "KVCacheStore is initialized.";
  config_ = config;
  kv_caches_ = kv_caches;
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

  rep_config_.replica_num = config_.replica_num;
  // rep_config_.preferred_segment = config_.localhost_name;

  if (!client_opt.has_value()) {
    LOG(FATAL) << "mooncake::Client::Create fail! Failed to create client with "
                  "host_name: "
               << config_.localhost_name;
  }
  client_ptr_ = client_opt.value();

  auto cache_cnt = 1;
  auto k_cache = kv_caches_->at(0).get_k_cache();
  if (config_.format == TensorFormat::LAYER_WISE) {
    k_cache_size_per_slice_ = k_cache[0].numel() * k_cache.element_size();
  } else {
    k_cache_size_per_slice_ = k_cache.numel() * k_cache.element_size();
  }
  LOG(INFO) << "k cache shape: " << k_cache.sizes()
            << ", size per slice: " << k_cache_size_per_slice_;

  auto v_cache = kv_caches_->at(0).get_v_cache();
  if (v_cache.defined() && v_cache.numel() != 0) {
    if (config_.format == TensorFormat::LAYER_WISE) {
      v_cache_size_per_slice_ = v_cache[0].numel() * v_cache.element_size();
    } else {
      v_cache_size_per_slice_ = v_cache.numel() * v_cache.element_size();
    }
    LOG(INFO) << "v cache shape: " << v_cache.sizes()
              << ", size per slice: " << v_cache_size_per_slice_;
    cache_cnt++;
  }

  auto index_cache = kv_caches_->at(0).get_index_cache();
  if (index_cache.defined() && index_cache.numel() != 0) {
    if (config_.format == TensorFormat::LAYER_WISE) {
      index_cache_size_per_slice_ =
          index_cache[0].numel() * index_cache.element_size();
    } else {
      index_cache_size_per_slice_ =
          index_cache.numel() * index_cache.element_size();
    }
    LOG(INFO) << "index cache shape: " << index_cache.sizes()
              << ", size per slice: " << index_cache_size_per_slice_;
    cache_cnt++;
  }

  if (config_.format == TensorFormat::LAYER_WISE) {
    LOG(INFO) << "slice cnt per block: " << cache_cnt * kv_caches_->size();

    layer_copy_index_pairs_.reserve(config_.layers_wise_copy_batchs);
    uint32_t layers_per_batch =
        kv_caches_->size() / config_.layers_wise_copy_batchs;
    for (uint32_t i = 0; i < config_.layers_wise_copy_batchs; i++) {
      uint32_t start_layer_idx = i * layers_per_batch;
      uint32_t end_layer_idx =
          std::min(start_layer_idx + layers_per_batch,
                   static_cast<uint32_t>(kv_caches_->size()));
      layer_copy_index_pairs_.emplace_back(
          std::make_pair(start_layer_idx, end_layer_idx));
    }
  } else {
    LOG(INFO) << "slice cnt per block: " << cache_cnt;
  }

  if (config_.protocol == "rdma") {
    if (config_.total_size > 0 && config_.tensor_data != nullptr) {
      if (!register_memory(config_.tensor_data, config_.total_size, "cpu:0")) {
        return false;
      }
    } else {
      LOG(ERROR) << "rdma must RegisterLocalMemory, but got register size: "
                 << config_.total_size
                 << ", and data ptr: " << uint64_t(config_.tensor_data);
      return false;
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

  str_keys.reserve(block_transfer_info.size() *
                   config_.layers_wise_copy_batchs);
  slices.reserve(block_transfer_info.size() * config_.layers_wise_copy_batchs);
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);
    str_key.append("-");
    str_key.append(std::to_string(config_.tp_rank));

    for (uint32_t i = 0; i < config_.layers_wise_copy_batchs; i++) {
      str_keys.emplace_back(str_key);
      str_keys.back().append("-");
      str_keys.back().append(std::to_string(i));
      auto exist = client_ptr_->IsExist(str_keys.back());
      if (exist.has_value() && exist.value()) {
        str_keys.pop_back();
        continue;
      }
      slices.emplace_back(
          std::move(genarate_mooncake_slice(block_info.src_block_id, i)));
    }
  }

  if (str_keys.empty()) {
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

uint32_t KVCacheStore::batch_get(Slice<BlockTransferInfo>& block_transfer_info,
                                 uint32_t slice_idx) {
  if (!is_initialized_) {
    return 0;
  }
  std::unordered_map<std::string, std::vector<mooncake::Slice>> slices;
  std::vector<std::string> str_keys;

  str_keys.reserve(block_transfer_info.size());
  for (auto block_info : block_transfer_info) {
    std::string str_key(reinterpret_cast<const char*>(block_info.hash_key),
                        MURMUR_HASH3_VALUE_LEN);

    str_key.append("-");
    str_key.append(std::to_string(config_.tp_rank));
    str_key.append("-");
    str_key.append(std::to_string(slice_idx));
    str_keys.emplace_back(str_key);
    slices.insert(std::make_pair(str_key,
                                 std::move(genarate_mooncake_slice(
                                     block_info.dst_block_id, slice_idx))));
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

uint32_t KVCacheStore::batch_exist(std::vector<std::string>& keys) {
  if (!is_initialized_) {
    return 0;
  }
  std::vector<std::string> query_keys;
  query_keys.reserve(keys.size() * config_.layers_wise_copy_batchs *
                     config_.tp_size);
  for (const auto& key : keys) {
    for (uint32_t i = 0; i < config_.tp_size; i++) {
      for (uint32_t j = 0; j < config_.layers_wise_copy_batchs; j++) {
        query_keys.emplace_back(key);
        query_keys.back().append("-");
        query_keys.back().append(std::to_string(i));
        query_keys.back().append("-");
        query_keys.back().append(std::to_string(j));
      }
    }
  }
  auto exist_vec = client_ptr_->BatchIsExist(std::move(query_keys));
  uint32_t ret = 0;
  for (auto exist : exist_vec) {
    if (!exist.has_value() || !exist.value()) {
      break;
    }
    ret++;
  }
  return ret / (config_.tp_size * config_.layers_wise_copy_batchs);
}

std::vector<mooncake::Slice> KVCacheStore::genarate_mooncake_slice(
    uint32_t block_id,
    uint32_t slice_idx) {
  switch (config_.format) {
    case TensorFormat::BLOCK_WISE: {
      std::vector<mooncake::Slice> slice;
      slice.reserve(3);

      void* k_cache = kv_caches_->at(block_id).get_k_cache().data_ptr();
      slice.emplace_back(mooncake::Slice{k_cache, k_cache_size_per_slice_});

      if (v_cache_size_per_slice_ != 0) {
        void* v_cache = kv_caches_->at(block_id).get_v_cache().data_ptr();
        slice.emplace_back(mooncake::Slice{v_cache, v_cache_size_per_slice_});
      }

      if (index_cache_size_per_slice_ != 0) {
        void* index_cache =
            kv_caches_->at(block_id).get_index_cache().data_ptr();
        slice.emplace_back(
            mooncake::Slice{index_cache, index_cache_size_per_slice_});
      }
      return slice;
    }
    case TensorFormat::LAYER_WISE: {
      auto [start_layer_idx, end_layer_idx] =
          layer_copy_index_pairs_[slice_idx];
      std::vector<mooncake::Slice> slice;
      slice.reserve((end_layer_idx - start_layer_idx) * 3);
      for (uint32_t i = start_layer_idx; i < end_layer_idx; i++) {
        void* k_cache = kv_caches_->at(i).get_k_cache()[block_id].data_ptr();
        slice.emplace_back(mooncake::Slice{k_cache, k_cache_size_per_slice_});

        if (v_cache_size_per_slice_ != 0) {
          void* v_cache = kv_caches_->at(i).get_v_cache()[block_id].data_ptr();
          slice.emplace_back(mooncake::Slice{v_cache, v_cache_size_per_slice_});
        }

        if (index_cache_size_per_slice_ != 0) {
          void* index_cache =
              kv_caches_->at(i).get_index_cache()[block_id].data_ptr();
          slice.emplace_back(
              mooncake::Slice{index_cache, index_cache_size_per_slice_});
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
