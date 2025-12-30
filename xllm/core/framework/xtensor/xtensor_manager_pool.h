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
#include <memory>

#include "framework/block/kv_cache_manager.h"
#include "xtensor_manager_client.h"
#include "xtensor_manager_server.h"

namespace xllm {
class XTensorManagerPool final : public KVCacheManager {
 public:
  explicit XTensorManagerPool(const xtensor::Options& options, int32_t dp_size);
  ~XTensorManagerPool();

  bool allocate(Sequence* sequence) override;
  bool allocate(std::vector<Sequence*>& sequences) override;
  bool allocate(Sequence* sequence, size_t num_tokens) override;

  void deallocate(Request* request) override;
  void deallocate(std::vector<Sequence*>& sequences) override;
  void deallocate(Sequence* sequence) override;

  std::vector<size_t> num_free_pages_per_layer() const;
  std::vector<size_t> num_used_pages_per_layer() const;
  double kv_cache_utilization() const override;

  // unimplemented functions
  void cache(Sequence* sequence) override {
    LOG(FATAL) << "cache is not implemented for page manager pool";
  }

  std::vector<Block> allocate(size_t num_tokens, int32_t& dp_rank) override {
    LOG(FATAL) << "allocate is not implemented for page manager pool";
    return {};
  }

  bool try_allocate(Sequence* sequence) override {
    LOG(FATAL) << "try_allocate is not implemented for page manager pool";
  }

  void allocate_shared(Sequence* sequence) override {
    LOG(FATAL) << "allocate_shared is not implemented for page manager pool";
  }

  std::vector<std::vector<BlockTransferInfo>>* get_swap_block_transfer_infos()
      override {
    LOG(FATAL) << "get_swap_block_transfer_infos is not implemented for page "
                  "manager pool";
    return nullptr;
  }

  void reset_transfer_infos() override {
    LOG(FATAL)
        << "reset_transfer_infos is not implemented for page manager pool";
  }

  uint32_t num_blocks() const override {
    LOG(FATAL) << "num_blocks is not implemented for page manager pool";
    return 0;
  }

  int32_t block_size() const override {
    LOG(FATAL) << "block_size is not implemented for page manager pool";
    return 0;
  }

  std::vector<size_t> num_blocks_in_prefix_cache() const override {
    LOG(FATAL) << "num_blocks_in_prefix_cache is not implemented for page "
                  "manager pool";
    return {};
  }

  std::vector<size_t> num_free_blocks() const override {
    LOG(FATAL) << "num_free_blocks is not implemented for page manager pool";
    return {};
  }

  std::vector<size_t> num_used_blocks() const override {
    LOG(FATAL) << "num_used_blocks is not implemented for page manager pool";
    return {};
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(XTensorManagerPool);
  void setup_single_node_xtensor_managers();
  void setup_multi_node_xtensor_managers(const std::string& master_node_addr);
  int32_t get_manager_with_max_free_pages() const;
  int32_t get_dp_rank(Sequence* sequence) const;

 private:
  xtensor::Options options_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  std::vector<std::shared_ptr<XTensorManagerClient>> xtensor_manager_clients_;
  std::vector<std::shared_ptr<XTensorManager>> xtensor_managers_;
  std::vector<std::unique_ptr<XTensorManagerServer>> xtensor_manager_servers_;
  std::string collective_server_name_;
};
}  // namespace xllm