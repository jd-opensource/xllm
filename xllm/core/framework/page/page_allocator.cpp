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

#include "page_allocator.h"

#include "common/global_flags.h"

namespace xllm {

PageAllocator::PageAllocator(const page::Options& options,
                             const torch::Device& device)
    : options_(options) {
  CHECK_GT(options_.num_total_pages(), 0) << "No pages to allocate";
  CHECK_EQ(options_.num_total_pages() % options_.num_layers(), 0)
      << "Total physical pages must be divisible by number of layers";

  num_free_phy_pages_per_layer_ =
      options_.num_total_pages() / options_.num_layers();

  free_phy_pages_.resize(options_.num_layers());
  for (auto& free_phy_pages_per_layer : free_phy_pages_) {
    free_phy_pages_per_layer.resize(num_free_phy_pages_per_layer_);
  }

  for (int64_t i = 0; i < options_.num_layers(); ++i) {
    for (int64_t j = num_free_phy_pages_per_layer_ - 1; j >= 0; --j)
      free_phy_pages_[i][j] = std::make_shared<PhyPage>(device);
  }

  free_phy_page_ids_.reserve(num_free_phy_pages_per_layer_);
  for (int64_t i = num_free_phy_pages_per_layer_ - 1; i >= 0; --i) {
    free_phy_page_ids_.push_back(i);
  }
}

std::vector<uint32_t> PageAllocator::allocate(int64_t n_pages_per_layer) {
  CHECK_LT(n_pages_per_layer, num_free_phy_pages_per_layer_)
      << "Not enough physical pages available";
  std::vector<uint32_t> phy_page_ids;
  phy_page_ids.resize(n_pages_per_layer);

  for (int64_t i = 0; i < n_pages_per_layer; ++i) {
    phy_page_ids[i] = allocate();
  }

  return phy_page_ids;
}

uint32_t PageAllocator::allocate() {
  CHECK_GT(num_free_phy_pages_per_layer_, 0)
      << "No more physical pages available";

  uint32_t phy_page_id;
  phy_page_id = free_phy_page_ids_[--num_free_phy_pages_per_layer_];

  return phy_page_id;
}

void PageAllocator::deallocate(std::vector<uint32_t>& page_ids) {
  for (auto& page_id : page_ids) {
    deallocate(page_id);
  }
}

// caller should make sure the page_id is valid
void PageAllocator::deallocate(uint32_t page_id) {
  CHECK_LT(num_free_phy_pages_per_layer_, free_phy_page_ids_.size());
  free_phy_page_ids_[num_free_phy_pages_per_layer_++] = page_id;
}

// map one virtual pointer to one physical page
void PageAllocator::map(VirPtr vir_ptr, PhyMemHandle phy_handle) const {
  VmmResult status;
#if defined(USE_NPU)
  status = aclrtMapMem(vir_ptr, FLAGS_granularity_size, 0, phy_handle, 0);
  CHECK_EQ(status, VmmSuccess)
      << "Failed to map virtual address to physical address";
#endif
}

void PageAllocator::map(VirPtr vir_ptr,
                        uint32_t page_id,
                        int64_t layer_idx) const {
  PhyMemHandle phy_handle =
      free_phy_pages_[layer_idx][page_id]->get_phy_handle();
  map(vir_ptr, phy_handle);
}

void PageAllocator::batch_map(VirPtr vir_ptr,
                              std::vector<uint32_t>& page_ids,
                              uint32_t num_new_pages,
                              int64_t layer_idx) const {
  size_t num_pages = page_ids.size();

  size_t ptr_offset = (num_pages - num_new_pages) * FLAGS_granularity_size;

  VirPtr temp_vir_ptr = reinterpret_cast<VirPtr>((char*)vir_ptr + ptr_offset);

  for (size_t j = num_new_pages; j > 0; --j) {
    uint32_t page_id = page_ids[num_pages - j];
    map(temp_vir_ptr, page_id, layer_idx);
    temp_vir_ptr =
        reinterpret_cast<VirPtr>((char*)temp_vir_ptr + FLAGS_granularity_size);
  }
}
}  // namespace xllm
