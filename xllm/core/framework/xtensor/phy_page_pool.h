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
#include <unordered_set>

#include "common/macros.h"
#include "options.h"
#include "phy_page.h"

namespace xllm {

// PhyPagePool is used to track memory pages of key and value. It is not
// thread safe. This class manages the allocation and deallocation of page.
class PhyPagePool final {
 public:
  PhyPagePool(const xtensor::Options& options, const torch::Device& device);

  ~PhyPagePool() = default;

  // allocate a list of page_ids for key or value for all layers
  std::vector<uint32_t> allocate(int64_t n_pages_per_layer);

  // allocate a page id for key or value for all layers
  uint32_t allocate();

  // get back one page to phy_page_pool
  void deallocate(uint32_t page_id);

  // get back a list of pages to phy_page_pool
  void deallocate(std::vector<uint32_t>& page_ids);

#if defined(USE_NPU)
  void map(VirPtr vir_ptr, PhyMemHandle phy_handle) const;
  void map(VirPtr vir_ptr, uint32_t page_id, int64_t layer_idx) const;
  void batch_map(VirPtr vir_ptr,
                 std::vector<uint32_t>& page_ids,
                 uint32_t num_new_pages,
                 int64_t layer_idx) const;
#endif

  // get num of total physical pages for key and value for all layers
  size_t get_num_total_phy_pages_per_layer() const {
    return free_phy_page_ids_.size();
  }

  // get num of free physical pages for key and value for one layer
  size_t get_num_free_phy_pages_per_layer() const {
    return num_free_phy_pages_per_layer_;
  }

  // get num of used physical pages for key and value for one layer
  size_t get_num_used_phy_pages_per_layer() const {
    return free_phy_page_ids_.size() - num_free_phy_pages_per_layer_;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(PhyPagePool);

 private:
  xtensor::Options options_;

  // free physical pages
  std::vector<std::vector<std::shared_ptr<PhyPage>>>
      free_phy_pages_;  // [num_layers, num_total_pages_per_layer]

  int64_t num_free_phy_pages_per_layer_;

  std::vector<uint32_t> free_phy_page_ids_;
};
}  // namespace xllm