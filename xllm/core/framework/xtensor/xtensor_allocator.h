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

#pragma once

#include <torch/torch.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "options.h"
#include "phy_page.h"
#include "xtensor.h"
#include "xtensor_dist_client.h"
#include "xtensor_dist_server.h"

namespace xllm {

/**
 * Per-model tensor storage
 */
struct ModelTensors {
  // K tensors: one tensor per layer (indexed by layer id)
  std::vector<std::unique_ptr<XTensor>> k_tensors;
  // V tensors: one tensor per layer (indexed by layer id)
  std::vector<std::unique_ptr<XTensor>> v_tensors;
  // Weight tensor (one large tensor for all model weights)
  std::unique_ptr<XTensor> weight_tensor;

  int64_t num_layers = 0;
  size_t kv_tensor_size_per_layer = 0;
  // Weight tensor bump allocator offset
  size_t weight_offset = 0;
  // Number of pages for weight tensor
  int64_t weight_num_pages = 0;
};

/**
 * XTensorAllocator manages XTensor objects for KV cache and model weights.
 *
 * This is a singleton class that:
 * - Creates and manages XTensor objects per model (indexed by model_id)
 * - Handles distributed XTensor operations via RPC
 * - Coordinates PhyPagePool initialization across workers
 */
class XTensorAllocator {
 public:
  // Get the global singleton instance
  static XTensorAllocator& get_instance() {
    static XTensorAllocator instance;
    return instance;
  }

  // Initialize the allocator with device configuration
  void init(const torch::Device& device);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // ============== KV Cache Interfaces ==============

  // Create K tensors for all layers of a model
  std::vector<torch::Tensor> create_k_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);

  // Create V tensors for all layers of a model
  std::vector<torch::Tensor> create_v_tensors(const std::string& model_id,
                                              const std::vector<int64_t>& dims,
                                              torch::Dtype dtype,
                                              int64_t num_layers);

  // KV tensor operations (partial mapping by offsets)
  bool map_to_kv_tensors(const std::string& model_id,
                         const std::vector<offset_t>& offsets);
  bool unmap_from_kv_tensors(const std::string& model_id,
                             const std::vector<offset_t>& offsets);

  // ============== Weight Tensor Interfaces ==============

  // Create weight tensor without mapping physical pages (for LIGHT_SLEEP mode)
  bool create_weight_tensor(const std::string& model_id, int64_t num_pages);

  // Weight tensor operations (full tensor mapping)
  bool map_weight_tensor(const std::string& model_id, int64_t num_pages);
  bool unmap_weight_tensor(const std::string& model_id);

  // Allocate a portion of the weight tensor for a specific layer/module
  bool allocate_weight(const std::string& model_id, void*& ptr, size_t size);

  // ============== Multi-node Setup ==============

  // Multi-node XTensor dist setup (called by rank0 to connect to other workers)
  void setup_multi_node_xtensor_dist(const xtensor::Options& options,
                                     const std::string& master_node_addr,
                                     int32_t dp_size);

  // Initialize PhyPagePool on all workers
  int64_t init_phy_page_pools(double max_memory_utilization = 0.9,
                              int64_t max_cache_size = 0);

  // ============== Broadcast Operations ==============

  // Broadcast KV tensor map/unmap to workers in a specific DP group
  bool broadcast_map_to_kv_tensors(const std::string& model_id,
                                   int32_t dp_rank,
                                   const std::vector<offset_t>& offsets);
  bool broadcast_unmap_from_kv_tensors(const std::string& model_id,
                                       int32_t dp_rank,
                                       const std::vector<offset_t>& offsets);

  // Broadcast weight tensor create/map/unmap to all workers
  bool broadcast_create_weight_tensor(const std::string& model_id,
                                      int64_t num_pages);
  bool broadcast_map_weight_tensor(const std::string& model_id,
                                   int64_t num_pages);
  bool broadcast_unmap_weight_tensor(const std::string& model_id);

  // Get XTensor dist clients (for distributed operations)
  const std::vector<std::shared_ptr<XTensorDistClient>>&
  get_xtensor_dist_clients() const {
    return xtensor_dist_clients_;
  }

  // Get device
  const torch::Device& device() const { return dev_; }

 private:
  XTensorAllocator() = default;
  ~XTensorAllocator();
  XTensorAllocator(const XTensorAllocator&) = delete;
  XTensorAllocator& operator=(const XTensorAllocator&) = delete;

  // Get or create model tensors (auto-creates if not exists)
  ModelTensors& get_or_create_model_tensors(const std::string& model_id);

  // Get model tensors (returns nullptr if not found)
  ModelTensors* get_model_tensors(const std::string& model_id);

  // Create K/V tensors implementation (handles lock and validation)
  // name: "K", "V", or future types like "index"
  std::vector<torch::Tensor> create_kv_tensors_impl_(
      const std::string& model_id,
      const std::vector<int64_t>& dims,
      torch::Dtype dtype,
      int64_t num_layers,
      const char* name);

  // Create tensors internal (must call with lock held)
  std::vector<torch::Tensor> create_tensors_internal_(
      size_t size,
      const std::vector<int64_t>& dims,
      torch::Dtype dtype,
      int64_t num_layers,
      std::vector<std::unique_ptr<XTensor>>& tensors_out);

  // Device initialization (platform-agnostic)
  void init_device_();

  // Cleanup resources
  void destroy();

  bool initialized_ = false;
  torch::Device dev_{torch::kCPU};

  mutable std::mutex mtx_;

  // Per-model tensors storage (key: model_id)
  std::unordered_map<std::string, ModelTensors> model_tensors_;

  // Zero page pointer (owned by PhyPagePool, not this class)
  PhyPage* zero_page_ = nullptr;

  // Multi-node XTensor dist members
  int32_t world_size_ = 0;  // total workers = dp_size * tp_size
  int32_t dp_size_ = 1;
  int32_t tp_size_ = 1;
  // DP group to worker clients mapping: dp_group_clients_[dp_rank][tp_rank]
  std::vector<std::vector<std::shared_ptr<XTensorDistClient>>>
      dp_group_clients_;
  // Flat list for backward compatibility and weight tensor broadcast
  std::vector<std::shared_ptr<XTensorDistClient>> xtensor_dist_clients_;
  std::vector<std::unique_ptr<XTensorDistServer>> xtensor_dist_servers_;
  std::string collective_server_name_{"XTensorAllocatorCollectiveServer"};
};

}  // namespace xllm
