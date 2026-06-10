/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "framework/kv_cache_transfer/kv_cache_store.h"

namespace xllm {
namespace {

constexpr uint32_t TIMEOUT_MS = 60000;
constexpr std::array<PrefixCacheGroup, 4> kAllPrefixCacheGroups = {
    PrefixCacheGroup::C1,
    PrefixCacheGroup::SINGLE,
    PrefixCacheGroup::C4,
    PrefixCacheGroup::C128,
};

std::vector<HierarchyKVCacheTransfer::LayerBatchRange> build_layer_batch_ranges(
    int64_t num_layers,
    uint32_t requested_batches) {
  std::vector<HierarchyKVCacheTransfer::LayerBatchRange> ranges;
  if (num_layers <= 0) {
    return ranges;
  }

  uint32_t layers_per_batch =
      requested_batches == 0
          ? static_cast<uint32_t>(num_layers)
          : static_cast<uint32_t>(num_layers) / requested_batches;
  layers_per_batch = std::max<uint32_t>(layers_per_batch, 1);

  for (int64_t begin = 0; begin < num_layers; begin += layers_per_batch) {
    ranges.push_back(
        {begin, std::min<int64_t>(begin + layers_per_batch, num_layers)});
  }
  return ranges;
}

torch::Tensor select_host_layer_slot(const torch::Tensor& host_tensor,
                                     int64_t layer_slot) {
  CHECK(host_tensor.defined()) << "host tensor must be defined.";
  CHECK_GT(host_tensor.dim(), 0) << "host tensor dim must be > 0.";
  CHECK_GE(layer_slot, 0) << "layer_slot must be non-negative.";
  CHECK_LT(layer_slot, host_tensor.size(0))
      << "layer_slot out of range, layer_slot=" << layer_slot
      << ", dim0=" << host_tensor.size(0);
  return host_tensor[layer_slot];
}

bool has_tensor(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0;
}

void emplace_prefix_tensor(PrefixCacheTensorMap* tensor_map,
                           KVCacheTensorRole::Value role,
                           const torch::Tensor& tensor) {
  CHECK(tensor_map != nullptr) << "tensor_map must not be null.";
  if (has_tensor(tensor)) {
    tensor_map->emplace(role, tensor);
  }
}

PrefixCacheTensorMap build_prefix_tensor_map(const KVCache& kv_cache,
                                             PrefixCacheGroup group) {
  PrefixCacheTensorMap tensor_map;

  const torch::Tensor key_cache = kv_cache.get_k_cache();
  const torch::Tensor value_cache = kv_cache.get_v_cache();
  const torch::Tensor index_cache = kv_cache.get_index_cache();
  const torch::Tensor conv_cache = kv_cache.get_conv_cache();
  const torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  const torch::Tensor swa_cache = kv_cache.get_swa_cache();

  const bool has_key = has_tensor(key_cache);
  const bool has_value = has_tensor(value_cache);
  const bool has_index = has_tensor(index_cache);
  const bool has_conv = has_tensor(conv_cache);
  const bool has_ssm = has_tensor(ssm_cache);
  const bool has_swa = has_tensor(swa_cache);

  switch (group) {
    case PrefixCacheGroup::C1:
      if (has_conv || has_ssm || has_swa) {
        return tensor_map;
      }
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::KEY, key_cache);
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::VALUE, value_cache);
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::INDEX, index_cache);
      return tensor_map;
    case PrefixCacheGroup::SINGLE:
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::CONV, conv_cache);
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::SSM, ssm_cache);
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::SWA, swa_cache);
      return tensor_map;
    case PrefixCacheGroup::C4:
      if (!has_swa || has_value || !has_key || !has_index) {
        return tensor_map;
      }
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::KEY, key_cache);
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::INDEX, index_cache);
      return tensor_map;
    case PrefixCacheGroup::C128:
      if (!has_swa || has_value || !has_key || has_index) {
        return tensor_map;
      }
      emplace_prefix_tensor(&tensor_map, KVCacheTensorRole::KEY, key_cache);
      return tensor_map;
    default:
      return tensor_map;
  }
}

}  // namespace

HierarchyKVCacheTransfer::HierarchyKVCacheTransfer(
    const Options& options,
    const torch::Device& device,
    std::vector<xllm::KVCache>* kv_caches_ptr,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : options_(options),
      device_(device),
      kv_caches_ptr_(kv_caches_ptr),
      kv_cache_shape_(kv_cache_shape),
      create_options_(create_options) {
  CHECK(kv_caches_ptr_ != nullptr) << "kv_caches_ptr must not be null.";

  device_.set_device();
  device_.init_device_context();
  load_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/2,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.h2d");
  offload_threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/5,
      /*init_func=*/[this]() mutable { device_.set_device(); },
      /*cpu_binding=*/false,
      /*pool_name=*/"HierarchyKVCacheTransfer.d2h");
  for (int i = 0; i < load_threadpool_->size() + offload_threadpool_->size();
       ++i) {
    copy_stream_.enqueue(device_.get_stream_from_pool(TIMEOUT_MS));
  }

  build_device_prefix_cache_map();
  layer_batch_ranges_ = build_layer_batch_ranges(
      options_.layers(), options_.layers_wise_copy_batchs());

  const bool requires_host_staging =
      options_.host_blocks_factor() > 1.0 || options_.enable_kvcache_store();
  if (requires_host_staging) {
    CHECK_GT(options_.host_blocks_factor(), 1.0)
        << "HierarchyKVCacheTransfer currently requires host staging when "
           "KVCacheStore is enabled.";
    batch_memcpy_ = create_batch_memcpy(device_);
    create_host_prefix_cache();
  }

  if (options_.enable_kvcache_store()) {
    CHECK(!host_kv_caches_.empty())
        << "KVCacheStore currently requires host prefix caches.";
    KVCacheStoreInitConfig config;
    config.localhost_name = options_.store_local_hostname();
    config.protocol = options_.store_protocol();
    config.metadata_server = options_.store_metadata_server();
    config.master_server_address = options_.store_master_server_address();
    config.tp_rank = options_.enable_mla() ? 0 : options_.tp_rank();
    config.tp_size = options_.enable_mla() ? 1 : options_.tp_size();
    StoreGroupedPrefixCaches store_caches;
    for (const auto& item : host_kv_caches_) {
      store_caches[item.first].push_back(item.second.get());
    }
    if (!KVCacheStore::get_instance().init(config, store_caches)) {
      LOG(FATAL) << "Init KVCacheStore fail!";
    }
  }
}

void HierarchyKVCacheTransfer::build_device_prefix_cache_map() {
  device_kv_caches_.clear();
  device_group_layer_ids_.clear();

  for (int64_t layer_id = 0;
       layer_id < static_cast<int64_t>(kv_caches_ptr_->size());
       ++layer_id) {
    KVCache& kv_cache = kv_caches_ptr_->at(static_cast<size_t>(layer_id));
    for (PrefixCacheGroup group : kAllPrefixCacheGroups) {
      PrefixCacheTensorMap tensor_map =
          build_prefix_tensor_map(kv_cache, group);
      if (!tensor_map.empty()) {
        device_kv_caches_[static_cast<PrefixCacheGroup::Value>(group)]
            .push_back(&kv_cache);
        device_group_layer_ids_[static_cast<PrefixCacheGroup::Value>(group)]
            .push_back(layer_id);
      }
    }
  }
}

std::vector<PrefixCacheGroup> HierarchyKVCacheTransfer::resolve_groups(
    PrefixCacheGroup group) const {
  std::vector<PrefixCacheGroup> groups;
  if (group == PrefixCacheGroup::INVALID) {
    groups.reserve(kAllPrefixCacheGroups.size());
    for (PrefixCacheGroup candidate : kAllPrefixCacheGroups) {
      if (device_kv_caches_.find(static_cast<PrefixCacheGroup::Value>(
              candidate)) != device_kv_caches_.end()) {
        groups.emplace_back(candidate);
      }
    }
    return groups;
  }

  if (device_kv_caches_.find(static_cast<PrefixCacheGroup::Value>(group)) !=
      device_kv_caches_.end()) {
    groups.emplace_back(group);
  }
  return groups;
}

void HierarchyKVCacheTransfer::create_host_prefix_cache() {
  CHECK(!device_kv_caches_.empty())
      << "device prefix caches must not be empty.";

  for (const auto& item : device_kv_caches_) {
    const PrefixCacheGroup group(item.first);
    const auto& group_caches = item.second;
    const auto& layer_ids = device_group_layer_ids_.at(item.first);
    CHECK_EQ(group_caches.size(), layer_ids.size())
        << "group caches size must match layer id size.";

    PrefixCacheTensorMap first_group_tensors =
        build_prefix_tensor_map(*group_caches.front(), group);
    CHECK(!first_group_tensors.empty()) << "group tensors must not be empty.";
    const int64_t group_layer_count = static_cast<int64_t>(group_caches.size());
    KVCacheCreateOptions host_create_options = create_options_;
    host_create_options.device(torch::Device(torch::kCPU))
        .enable_xtensor(false)
        .enable_raw_device_allocator(false)
        .host_blocks_factor(options_.host_blocks_factor())
        .enable_kv_cache_huge_page_allocator(false);
    host_kv_caches_[item.first] =
        std::make_unique<KVCache>(kv_cache_shape_, host_create_options, group);

    const std::vector<std::vector<int64_t>> host_shapes =
        host_kv_caches_[item.first]->get_shapes();
    for (const std::vector<int64_t>& shape : host_shapes) {
      if (shape.empty()) {
        continue;
      }
      CHECK_GT(shape.size(), static_cast<size_t>(1))
          << "host prefix cache tensor must contain a layer dimension.";
      CHECK_EQ(shape[1], group_layer_count)
          << "host prefix cache layer count mismatch for group "
          << group.to_string();
    }
  }
}

HierarchyKVCacheTransfer::PrefixCacheCopyPlan
HierarchyKVCacheTransfer::build_copy_plan(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    const LayerBatchRange& layer_batch_range) const {
  PrefixCacheCopyPlan plan;
  CHECK(!block_transfer_info.empty())
      << "block_transfer_info must not be empty.";

  const TransferType transfer_type = block_transfer_info.front().transfer_type;
  for (const auto& info : block_transfer_info) {
    for (PrefixCacheGroup group : resolve_groups(info.group)) {
      const auto device_group_it =
          device_kv_caches_.find(static_cast<PrefixCacheGroup::Value>(group));
      const auto layer_ids_it = device_group_layer_ids_.find(
          static_cast<PrefixCacheGroup::Value>(group));
      const auto host_group_it =
          host_kv_caches_.find(static_cast<PrefixCacheGroup::Value>(group));
      CHECK(device_group_it != device_kv_caches_.end())
          << "Missing device group for " << group.to_string();
      CHECK(layer_ids_it != device_group_layer_ids_.end())
          << "Missing device layer ids for " << group.to_string();
      CHECK(host_group_it != host_kv_caches_.end())
          << "Missing host group for " << group.to_string();

      const auto& group_caches = device_group_it->second;
      const auto& layer_ids = layer_ids_it->second;
      const KVCache* host_group_cache = host_group_it->second.get();
      CHECK(host_group_cache != nullptr)
          << "Missing host cache instance for " << group.to_string();
      int32_t host_block_id = -1;
      int32_t device_block_id = -1;
      switch (transfer_type) {
        case TransferType::H2D:
          CHECK_GE(info.src_block_id, 0)
              << "Host-staging load requires src_block_id >= 0.";
          CHECK_GE(info.dst_block_id, 0)
              << "Host-staging load requires dst_block_id >= 0.";
          host_block_id = info.src_block_id;
          device_block_id = info.dst_block_id;
          break;
        case TransferType::D2H2G:
          CHECK_GE(info.src_block_id, 0)
              << "Host-staging offload requires src_block_id >= 0.";
          CHECK_GE(info.dst_block_id, 0)
              << "Host-staging offload requires dst_block_id >= 0.";
          host_block_id = info.dst_block_id;
          device_block_id = info.src_block_id;
          break;
        case TransferType::D2G:
        default:
          LOG(FATAL) << "Unsupported transfer type for copy plan: "
                     << static_cast<uint32_t>(transfer_type);
      }
      CHECK_GE(host_block_id, 0) << "host block id must be non-negative.";
      PrefixCacheTensorMap host_block =
          host_group_cache->get_prefix_cache_tensors(group, host_block_id);
      CHECK(!host_block.empty())
          << "host prefix cache block is empty, group=" << group.to_string()
          << ", host_block_id=" << host_block_id;
      for (size_t layer_slot = 0; layer_slot < group_caches.size();
           ++layer_slot) {
        const int64_t absolute_layer_id = layer_ids[layer_slot];
        if (absolute_layer_id < layer_batch_range.begin_layer ||
            absolute_layer_id >= layer_batch_range.end_layer) {
          continue;
        }

        PrefixCacheTensorMap device_rows =
            group_caches[layer_slot]->get_prefix_cache_tensors(group,
                                                               device_block_id);
        for (const auto& device_item : device_rows) {
          const auto host_it = host_block.find(device_item.first);
          CHECK(host_it != host_block.end())
              << "missing host tensor for role "
              << static_cast<int32_t>(device_item.first);
          torch::Tensor host_row = select_host_layer_slot(
              host_it->second, static_cast<int64_t>(layer_slot));
          if (transfer_type == TransferType::H2D) {
            plan.src_tensors.emplace_back(host_row);
            plan.dst_tensors.emplace_back(device_item.second);
          } else {
            plan.src_tensors.emplace_back(device_item.second);
            plan.dst_tensors.emplace_back(host_row);
          }
        }
      }
    }
  }

  return plan;
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());

  switch (block_transfer_info[0].transfer_type) {
    case TransferType::D2H2G:
      return offload(block_transfer_info);
    case TransferType::H2D: {
      load_threadpool_->schedule(
          [this,
           batch_id,
           block_transfer_info = std::move(block_transfer_info)]() mutable {
            load_from_host(batch_id, block_transfer_info);
          });
      return 0;
    }
    case TransferType::G2H:
      return KVCacheStore::get_instance().batch_get(block_transfer_info);
    default:
      LOG(ERROR) << "Unsupport copy type: "
                 << static_cast<uint32_t>(block_transfer_info[0].transfer_type);
      return 0;
  }
}

uint32_t HierarchyKVCacheTransfer::transfer_kv_blocks(
    const uint64_t /*batch_id*/,
    Slice<BlockTransferInfo>& block_transfer_info) {
  CHECK(!block_transfer_info.empty());
  switch (block_transfer_info[0].transfer_type) {
    case TransferType::G2H:
      return KVCacheStore::get_instance().batch_get(block_transfer_info);
    default:
      LOG(ERROR) << "Unsupport copy type: "
                 << static_cast<uint32_t>(block_transfer_info[0].transfer_type);
      return 0;
  }
}

void HierarchyKVCacheTransfer::set_layer_synchronizer(
    ModelInputParams& params) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = layer_wise_load_synchronizer_.find(params.meta.batch_id);
  if (it == layer_wise_load_synchronizer_.end()) {
    return;
  }
  auto event_cnt = it->second->size();
  auto layers_per_bacth_copy = (options_.layers() + event_cnt - 1) / event_cnt;
  params.parallel.layer_wise_load_synchronizer = it->second;
  params.parallel.layers_per_bacth_copy = layers_per_bacth_copy;
  layer_wise_load_synchronizer_.erase(it);
}

uint32_t HierarchyKVCacheTransfer::offload(
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return 0;
  }

  Slice<BlockTransferInfo> slice(block_transfer_info);
  if (!host_kv_caches_.empty()) {
    if (!offload_to_host(slice)) {
      LOG(ERROR) << "Offload to host fail!";
      return 0;
    }
  }

  if (options_.enable_kvcache_store() &&
      (!options_.enable_mla() || options_.tp_rank() == 0)) {
    const uint32_t success_cnt = KVCacheStore::get_instance().batch_put(slice);
    if (success_cnt != slice.size()) {
      LOG(WARNING) << "KVCacheStore not all put success: " << success_cnt << "/"
                   << slice.size();
    }
  }

  return block_transfer_info.size();
}

bool HierarchyKVCacheTransfer::offload_to_host(
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return true;
  }

  CHECK(!host_kv_caches_.empty()) << "host prefix caches must be initialized.";
  CHECK(batch_memcpy_ != nullptr) << "batch memcpy must be initialized.";
  std::unique_ptr<Stream> stream;
  copy_stream_.wait_dequeue(stream);
  bool success = true;
  for (const auto& range : layer_batch_ranges_) {
    PrefixCacheCopyPlan plan = build_copy_plan(
        static_cast<std::vector<BlockTransferInfo>>(block_transfer_info),
        range);
    if (!batch_memcpy_->copy_d2h(
            plan.src_tensors, plan.dst_tensors, stream.get())) {
      success = false;
      break;
    }
  }
  copy_stream_.enqueue(std::move(stream));
  return success;
}

bool HierarchyKVCacheTransfer::load_from_host(
    uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (block_transfer_info.empty()) {
    return true;
  }

  auto synchronizer = create_layer_synchronizer(
      static_cast<int64_t>(layer_batch_ranges_.size()));
  CHECK(synchronizer != nullptr) << "layer synchronizer must not be null.";
  CHECK_EQ(synchronizer->size(), layer_batch_ranges_.size())
      << "layer synchronizer size mismatch.";
  CHECK(batch_memcpy_ != nullptr) << "batch memcpy must be initialized.";
  {
    std::lock_guard<std::mutex> lock(mutex_);
    layer_wise_load_synchronizer_[batch_id] = synchronizer;
  }

  std::unique_ptr<Stream> stream;
  copy_stream_.wait_dequeue(stream);
  bool success = true;
  for (size_t range_idx = 0; range_idx < layer_batch_ranges_.size();
       ++range_idx) {
    PrefixCacheCopyPlan plan =
        build_copy_plan(block_transfer_info, layer_batch_ranges_[range_idx]);
    if (!batch_memcpy_->copy_h2d(
            plan.src_tensors, plan.dst_tensors, stream.get())) {
      success = false;
      break;
    }
    if (!synchronizer->record_stream(static_cast<int64_t>(range_idx),
                                     stream.get())) {
      success = false;
      break;
    }
  }

  copy_stream_.enqueue(std::move(stream));
  return success;
}

}  // namespace xllm
