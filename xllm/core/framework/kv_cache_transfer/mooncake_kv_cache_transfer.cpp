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

#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"

#include <glog/logging.h>

#include <numeric>

#if defined(USE_NPU)
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#endif

#include "common/global_flags.h"
#include "framework/xtensor/global_xtensor.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "util/net.h"

namespace xllm {

// ============================================================================
// MooncakeKVCacheTransferBase
// ============================================================================

MooncakeKVCacheTransferBase::MooncakeKVCacheTransferBase(
    const int32_t device_id,
    const int16_t listen_port,
    const torch::Device& device,
    std::unique_ptr<MooncakeTransferEngine> engine)
    : device_id_(device_id),
      device_(device),
      listen_port_(listen_port),
      mooncake_te_(std::move(engine)) {
  std::string instance_ip = net::get_local_ip_addr();
  cluster_id_ = net::convert_ip_port_to_uint64(instance_ip, listen_port_);
}

void MooncakeKVCacheTransferBase::initialize(int32_t device_id) {
  (void)device_id;
  addr_ = mooncake_te_->initialize();
}

void MooncakeKVCacheTransferBase::get_cache_info(uint64_t& cluster_id,
                                                 std::string& addr,
                                                 int64_t& key_cache_id,
                                                 int64_t& value_cache_id) {
  cluster_id = cluster_id_;
  addr = addr_;
  key_cache_id = 0;
  value_cache_id = 0;

  LOG(INFO) << "get_cache_info success, cluster_id=" << cluster_id_
            << ", addr=" << addr_;
}

bool MooncakeKVCacheTransferBase::link_cluster(const uint64_t cluster_id,
                                               const std::string& remote_addr,
                                               const std::string& device_ip,
                                               const uint16_t port) {
  LOG(INFO) << "link_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->open_session(cluster_id, remote_addr);
}

bool MooncakeKVCacheTransferBase::unlink_cluster(const uint64_t& cluster_id,
                                                 const std::string& remote_addr,
                                                 const std::string& device_ip,
                                                 const uint16_t port,
                                                 bool force_flag) {
  LOG(INFO) << "unlink_cluster, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  return mooncake_te_->close_session(cluster_id, remote_addr);
}

// ============================================================================
// MooncakeKVCacheTransferDefault
// ============================================================================

MooncakeKVCacheTransferDefault::MooncakeKVCacheTransferDefault(
    const int32_t device_id,
    const int16_t listen_port,
    const torch::Device& device,
    const std::string& model_type)
    : MooncakeKVCacheTransferBase(
          device_id,
          listen_port,
          device,
          std::make_unique<MooncakeTransferEngine>(listen_port, device)),
      model_type_(model_type) {}

void MooncakeKVCacheTransferDefault::allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = num_layers;
  allocate_kv_cache_impl(kv_caches, num_layers, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferDefault::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = kv_caches.size();
  if (!kv_caches.empty()) {
    torch::Tensor value_cache = kv_caches[0].get_v_cache();
    torch::Tensor index_cache = kv_caches[0].get_index_cache();
    has_v_cache_ = value_cache.defined() && value_cache.numel() > 0;
    has_index_cache_ = index_cache.defined() && index_cache.numel() > 0;
  }
  buf_cnt_per_layer_ = 1 + static_cast<int64_t>(has_v_cache_) +
                       static_cast<int64_t>(has_index_cache_);

  int64_t data_size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  int64_t count_per_block = 1;
  for (size_t i = 1; i < kv_cache_shape[0].size(); ++i) {
    count_per_block *= kv_cache_shape[0][i];
  }
  size_per_block_ = count_per_block * data_size;

  register_kv_cache_impl(kv_caches);
}

void MooncakeKVCacheTransferDefault::allocate_kv_cache_impl(
    std::vector<xllm::KVCache>& kv_caches,
    int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
#if defined(USE_MLU)
  torch::TensorOptions options =
      torch::TensorOptions().dtype(dtype).device(device_);
  for (int64_t i = 0; i < num_layers; ++i) {
    torch::Tensor key_cache = torch::zeros(kv_cache_shape[0], options);
    torch::Tensor value_cache;
    torch::Tensor index_cache;
    if (kv_cache_shape.size() > 1 && !kv_cache_shape[1].empty()) {
      value_cache = torch::zeros(kv_cache_shape[1], options);
    }
    if (kv_cache_shape.size() > 2 && !kv_cache_shape[2].empty()) {
      index_cache = torch::zeros(kv_cache_shape[2], options);
    }
    if (index_cache.defined()) {
      kv_caches.emplace_back(key_cache, value_cache, index_cache);
    } else {
      kv_caches.emplace_back(key_cache, value_cache);
    }
  }
#else
  // Original mode: allocate device memory using aclrtMalloc
  // calculate the size of kv cache for each layer
  auto data_size = torch::elementSize(dtype);
  int64_t k_cache_size_per_layer = data_size;
  for (int64_t i = 0; i < kv_cache_shape[0].size(); ++i) {
    k_cache_size_per_layer *= kv_cache_shape[0][i];
  }
  int64_t v_cache_size_per_layer = data_size;
  for (int64_t i = 0; i < kv_cache_shape[1].size(); ++i) {
    v_cache_size_per_layer *= kv_cache_shape[1][i];
  }

  // allocate device memory for kv cache
  std::vector<uint64_t> k_cache_addrs;
  std::vector<uint64_t> v_cache_addrs;
  k_cache_addrs.reserve(num_layers);
  v_cache_addrs.reserve(num_layers);

  std::vector<uintptr_t> k_tensor_addrs;
  std::vector<uintptr_t> v_tensor_addrs;
  k_tensor_addrs.reserve(num_layers);
  v_tensor_addrs.reserve(num_layers);
  for (int64_t i = 0; i < num_layers; ++i) {
    void* k_cache_buffer = nullptr;
    void* v_cache_buffer = nullptr;
    auto acl_ret = aclrtMalloc(
        &k_cache_buffer, k_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK(acl_ret == ACL_SUCCESS) << "aclrtMalloc k cache failed.";
    acl_ret = aclrtMalloc(
        &v_cache_buffer, v_cache_size_per_layer, ACL_MEM_MALLOC_HUGE_ONLY);
    CHECK(acl_ret == ACL_SUCCESS) << "aclrtMalloc v cache failed.";

    k_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(k_cache_buffer));
    v_cache_addrs.emplace_back(reinterpret_cast<uint64_t>(v_cache_buffer));

    k_tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(k_cache_buffer));
    v_tensor_addrs.emplace_back(reinterpret_cast<uintptr_t>(v_cache_buffer));
  }

  // convert memory addrs to torch tensors
  aclFormat npu_format_type =
      model_type_ == "deepseek_v3" && FLAGS_enable_prefix_cache
          ? ACL_FORMAT_FRACTAL_NZ
          : ACL_FORMAT_ND;
  auto k_torch_tensors = convert_to_torch_tensor(
      kv_cache_shape[0], dtype, k_tensor_addrs, npu_format_type);
  auto v_torch_tensors = convert_to_torch_tensor(
      kv_cache_shape[1], dtype, v_tensor_addrs, npu_format_type);

  torch::Tensor key_cache, value_cache;
  for (int64_t i = 0; i < num_layers; ++i) {
    key_cache = k_torch_tensors[i];
    value_cache = v_torch_tensors[i];
    kv_caches.emplace_back(key_cache, value_cache);
  }
#endif
}

void MooncakeKVCacheTransferDefault::add_buf(
    const torch::Tensor& tensor,
    std::vector<void*>& addrs,
    std::vector<size_t>& lens,
    std::vector<uint64_t>& buf_bytes) const {
  if (!tensor.defined() || tensor.numel() == 0) {
    return;
  }

  CHECK_GT(tensor.dim(), 0) << "cache tensor dim must be positive";
  int64_t block_cnt = tensor.size(0);
  CHECK_GT(block_cnt, 0) << "cache tensor block dim must be positive";

  addrs.emplace_back(tensor.data_ptr());
  lens.emplace_back(static_cast<size_t>(tensor.nbytes()));
  buf_bytes.emplace_back(static_cast<uint64_t>(tensor.nbytes() / block_cnt));
}

std::vector<int64_t> MooncakeKVCacheTransferDefault::get_buf_ids(
    const std::vector<int64_t>& layer_ids) const {
  std::vector<int64_t> active_layer_ids;
  if (layer_ids.empty()) {
    active_layer_ids.resize(static_cast<size_t>(num_layers_));
    std::iota(active_layer_ids.begin(), active_layer_ids.end(), 0);
  } else {
    active_layer_ids = layer_ids;
  }

  std::vector<int64_t> buf_ids;
  buf_ids.reserve(active_layer_ids.size() *
                  static_cast<size_t>(buf_cnt_per_layer_));
  for (int64_t layer_id : active_layer_ids) {
    CHECK_GE(layer_id, 0) << "layer_id must be non-negative";
    CHECK_LT(layer_id, num_layers_) << "layer_id out of range";

    int64_t buf_id = layer_id * buf_cnt_per_layer_;
    buf_ids.emplace_back(buf_id++);
    if (has_v_cache_) {
      buf_ids.emplace_back(buf_id++);
    }
    if (has_index_cache_) {
      buf_ids.emplace_back(buf_id);
    }
  }
  return buf_ids;
}

void MooncakeKVCacheTransferDefault::register_kv_cache_impl(
    std::vector<xllm::KVCache>& kv_caches) {
  std::vector<void*> addrs;
  std::vector<size_t> lens;
  std::vector<uint64_t> buf_bytes;
  addrs.reserve(static_cast<size_t>(num_layers_) * 3);
  lens.reserve(static_cast<size_t>(num_layers_) * 3);
  buf_bytes.reserve(static_cast<size_t>(num_layers_) * 3);

  for (int64_t i = 0; i < num_layers_; ++i) {
    add_buf(kv_caches[i].get_k_cache(), addrs, lens, buf_bytes);
    add_buf(kv_caches[i].get_v_cache(), addrs, lens, buf_bytes);
    add_buf(kv_caches[i].get_index_cache(), addrs, lens, buf_bytes);
  }

  if (!mooncake_te_->register_memory(addrs, lens, buf_bytes)) {
    LOG(ERROR) << "register_kv_cache_impl failed";
    return;
  }

  LOG(INFO) << "register_kv_cache_impl success, num_layers=" << num_layers_
            << ", buffers=" << buf_bytes.size();
}

bool MooncakeKVCacheTransferDefault::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  (void)src_cluster_id;
  (void)src_k_cache_id;
  (void)src_v_cache_id;
  std::vector<int64_t> layer_ids;
  std::vector<int64_t> buf_ids = get_buf_ids(layer_ids);
  auto ret = mooncake_te_->pull_memory_blocks(
      src_addr, src_blocks, dst_blocks, buf_ids);
  if (!ret) {
    LOG(ERROR) << "Pull kv cache blocks failed, ret = " << ret;
    return false;
  }
  return true;
}

bool MooncakeKVCacheTransferDefault::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  (void)is_spec_draft;
  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    layer_synchronizer->synchronize_layer(layer_index);
    for (const auto& pair : merged_kv_infos) {
      std::vector<int64_t> layer_ids = {layer_index};
      std::vector<int64_t> buf_ids = get_buf_ids(layer_ids);
      const KVCacheInfo& kv_info = pair.second;
      auto ret = mooncake_te_->push_memory_blocks(
          kv_info.dst_addr, kv_info.src_blocks, kv_info.dst_blocks, buf_ids);
      if (!ret) {
        LOG(ERROR) << "Push kv blocks failed, layer = " << layer_index
                   << ", ret = " << ret;
        return false;
      }
    }
  }
  return true;
}

// ============================================================================
// MooncakeKVCacheTransferXTensor
// ============================================================================

MooncakeKVCacheTransferXTensor::MooncakeKVCacheTransferXTensor(
    const int32_t device_id,
    const int16_t listen_port,
    const torch::Device& device)
    : MooncakeKVCacheTransferBase(
          device_id,
          listen_port,
          device,
          std::make_unique<MooncakeTransferEngine>(listen_port, device)) {}

void MooncakeKVCacheTransferXTensor::allocate_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = num_layers;
  allocate_kv_cache_impl(kv_caches, num_layers, kv_cache_shape, dtype);
}

void MooncakeKVCacheTransferXTensor::register_kv_cache(
    std::vector<xllm::KVCache>& kv_caches,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  num_layers_ = kv_caches.size();

  int64_t data_size = torch::scalarTypeToTypeMeta(dtype).itemsize();
  int64_t count_per_block = 1;
  for (int32_t i = 1; i < kv_cache_shape[0].size(); ++i) {
    count_per_block *= kv_cache_shape[0][i];
  }
  size_per_block_ = count_per_block * data_size;

  register_kv_cache_impl();
}

void MooncakeKVCacheTransferXTensor::allocate_kv_cache_impl(
    std::vector<xllm::KVCache>& kv_caches,
    int64_t num_layers,
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    torch::ScalarType dtype) {
  auto& allocator = XTensorAllocator::get_instance();
  CHECK(!model_id_.empty()) << "model_id must be set for XTensor mode";

  auto k_tensors = allocator.create_k_tensors(
      model_id_, kv_cache_shape[0], dtype, num_layers);
  auto v_tensors = allocator.create_v_tensors(
      model_id_, kv_cache_shape[1], dtype, num_layers);

  for (int64_t i = 0; i < num_layers; ++i) {
#if defined(USE_NPU)
    auto k_tensor =
        at_npu::native::npu_format_cast(k_tensors[i], ACL_FORMAT_ND);
    auto v_tensor =
        at_npu::native::npu_format_cast(v_tensors[i], ACL_FORMAT_ND);
    kv_caches.emplace_back(k_tensor, v_tensor);
#else
    kv_caches.emplace_back(k_tensors[i], v_tensors[i]);
#endif
  }

  LOG(INFO) << "MooncakeKVCacheTransferXTensor: KV cache allocated"
            << ", model_id=" << model_id_ << ", num_layers=" << num_layers;
}

void MooncakeKVCacheTransferXTensor::register_kv_cache_impl() {
  // XTensor mode registers one shared GlobalXTensor memory region.
  auto& global_xtensor = GlobalXTensor::get_instance();
  if (!global_xtensor.is_initialized()) {
    LOG(ERROR) << "GlobalXTensor not initialized in xtensor mode";
    return;
  }

  if (global_xtensor.is_mooncake_registered()) {
    LOG(INFO) << "GlobalXTensor already registered to mooncake, skip";
    return;
  }

  std::vector<void*> addrs = {global_xtensor.base_vaddr()};
  std::vector<size_t> lens = {global_xtensor.total_size()};
  std::vector<uint64_t> buf_bytes = {static_cast<uint64_t>(size_per_block_)};

  if (!mooncake_te_->register_memory(addrs, lens, buf_bytes)) {
    LOG(ERROR) << "register GlobalXTensor failed";
    return;
  }

  global_xtensor.set_mooncake_registered(true);
  LOG(INFO) << "register_kv_cache_impl success, total_size="
            << global_xtensor.total_size()
            << ", num_pages=" << global_xtensor.num_total_pages()
            << ", size_per_block=" << size_per_block_;
}

bool MooncakeKVCacheTransferXTensor::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const int64_t src_k_cache_id,
    const int64_t src_v_cache_id,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  (void)src_cluster_id;
  (void)src_k_cache_id;
  (void)src_v_cache_id;
  return pull_kv_blocks_impl(src_addr, src_blocks, dst_blocks);
}

bool MooncakeKVCacheTransferXTensor::push_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer,
    bool is_spec_draft) {
  (void)is_spec_draft;
  return push_kv_blocks_impl(merged_kv_infos, layer_synchronizer);
}

bool MooncakeKVCacheTransferXTensor::pull_kv_blocks_impl(
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks) {
  if (model_id_.empty()) {
    LOG(ERROR) << "model_id not set for XTensor mode pull";
    return false;
  }

  auto& allocator = XTensorAllocator::get_instance();

  // For each layer, convert block_ids to GlobalXTensor offsets and transfer
  for (int64_t layer_id = 0; layer_id < num_layers_; ++layer_id) {
    std::vector<uint64_t> src_offsets;
    std::vector<uint64_t> dst_offsets;
    src_offsets.reserve(src_blocks.size() * 2);  // K and V
    dst_offsets.reserve(dst_blocks.size() * 2);

    for (size_t i = 0; i < src_blocks.size(); ++i) {
      // Source block -> GlobalXTensor offsets
      auto [src_k_off, src_v_off] = allocator.get_global_offsets_for_block(
          model_id_, layer_id, src_blocks[i], size_per_block_);
      if (src_k_off == UINT64_MAX || src_v_off == UINT64_MAX) {
        LOG(ERROR) << "Failed to get source offsets for block " << src_blocks[i]
                   << " at layer " << layer_id;
        return false;
      }

      // Destination block -> GlobalXTensor offsets
      auto [dst_k_off, dst_v_off] = allocator.get_global_offsets_for_block(
          model_id_, layer_id, dst_blocks[i], size_per_block_);
      if (dst_k_off == UINT64_MAX || dst_v_off == UINT64_MAX) {
        LOG(ERROR) << "Failed to get dest offsets for block " << dst_blocks[i]
                   << " at layer " << layer_id;
        return false;
      }

      // K cache offsets
      src_offsets.push_back(src_k_off);
      dst_offsets.push_back(dst_k_off);
      // V cache offsets
      src_offsets.push_back(src_v_off);
      dst_offsets.push_back(dst_v_off);
    }

    auto* te = static_cast<MooncakeTransferEngine*>(mooncake_te_.get());
    auto ret = te->move_memory_by_global_offsets(
        src_addr,
        src_offsets,
        dst_offsets,
        size_per_block_,
        MooncakeTransferEngine::MoveOpcode::READ);
    if (!ret) {
      LOG(ERROR) << "pull_kv_blocks_impl failed at layer " << layer_id;
      return false;
    }
  }

  VLOG(1) << "pull_kv_blocks_impl success, num_blocks=" << src_blocks.size()
          << ", num_layers=" << num_layers_;
  return true;
}

bool MooncakeKVCacheTransferXTensor::push_kv_blocks_impl(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    std::shared_ptr<KVPushSynchronizerImpl>& layer_synchronizer) {
  if (model_id_.empty()) {
    LOG(ERROR) << "model_id not set for XTensor mode push";
    return false;
  }

  auto& allocator = XTensorAllocator::get_instance();

  for (int64_t layer_index = 0; layer_index < num_layers_; ++layer_index) {
    // Wait for the KV cache computation of this layer to complete.
    layer_synchronizer->synchronize_layer(layer_index);

    // Push the KV Cache computed at this layer for all requests
    for (const auto& pair : merged_kv_infos) {
      const KVCacheInfo& kv_info = pair.second;

      // Check if we have XTensor offsets from D-node
      bool has_dst_offsets = !kv_info.dst_xtensor_layer_offsets.empty() &&
                             static_cast<size_t>(layer_index) <
                                 kv_info.dst_xtensor_layer_offsets.size();

      std::vector<uint64_t> src_offsets;
      std::vector<uint64_t> dst_offsets;
      src_offsets.reserve(kv_info.src_blocks.size() * 2);
      dst_offsets.reserve(kv_info.dst_blocks.size() * 2);

      for (size_t i = 0; i < kv_info.src_blocks.size(); ++i) {
        // Source block -> GlobalXTensor offsets (calculate locally on P-node)
        auto [src_k_off, src_v_off] = allocator.get_global_offsets_for_block(
            model_id_, layer_index, kv_info.src_blocks[i], size_per_block_);
        if (src_k_off == UINT64_MAX || src_v_off == UINT64_MAX) {
          LOG(ERROR) << "Failed to get source offsets for block "
                     << kv_info.src_blocks[i] << " at layer " << layer_index;
          return false;
        }

        // Destination offsets: use offsets from D-node if available
        uint64_t dst_k_off, dst_v_off;
        if (has_dst_offsets) {
          const auto& layer_offsets =
              kv_info.dst_xtensor_layer_offsets[layer_index];
          if (i < layer_offsets.k_offsets.size() &&
              i < layer_offsets.v_offsets.size()) {
            dst_k_off = layer_offsets.k_offsets[i];
            dst_v_off = layer_offsets.v_offsets[i];
          } else {
            LOG(ERROR) << "XTensor offset index out of range for block " << i
                       << " at layer " << layer_index;
            return false;
          }
        } else {
          LOG(ERROR) << "No XTensor destination offsets from D-node for layer "
                     << layer_index;
          return false;
        }

        // K cache offsets
        src_offsets.push_back(src_k_off);
        dst_offsets.push_back(dst_k_off);
        // V cache offsets
        src_offsets.push_back(src_v_off);
        dst_offsets.push_back(dst_v_off);
      }
      auto* xtensor_te =
          static_cast<MooncakeTransferEngine*>(mooncake_te_.get());
      auto ret = xtensor_te->move_memory_by_global_offsets(
          kv_info.dst_addr,
          src_offsets,
          dst_offsets,
          size_per_block_,
          MooncakeTransferEngine::MoveOpcode::WRITE);
      if (!ret) {
        LOG(ERROR) << "push_kv_blocks_impl failed at layer " << layer_index;
        return false;
      }
    }
  }

  VLOG(1) << "push_kv_blocks_impl success, num_layers=" << num_layers_;
  return true;
}

}  // namespace xllm
