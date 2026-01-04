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

#include <memory>
#include <mutex>
#include <vector>

#include "phy_page.h"

namespace xllm {

/**
 * PhyPagePool manages a pool of pre-allocated physical pages.
 *
 * This is a singleton class that:
 * - Pre-allocates physical pages during initialization
 * - Provides get/put interface for XTensor to acquire/release physical pages
 * - Avoids runtime allocation overhead during map operations
 */
class PhyPagePool {
 public:
  // Get the global singleton instance
  static PhyPagePool& get_instance() {
    static PhyPagePool pool;
    return pool;
  }

  // Initialize the pool with specified number of pages
  // device: the device to allocate physical pages on
  // num_pages: number of physical pages to pre-allocate
  void init(const torch::Device& device, size_t num_pages);

  // Check if initialized
  bool is_initialized() const { return initialized_; }

  // Get a physical page from the pool
  // Returns nullptr if pool is empty
  std::unique_ptr<PhyPage> get();

  // Get multiple physical pages from the pool in one lock
  // Returns empty vector if not enough pages available
  // If partial allocation fails, all acquired pages are returned to pool
  std::vector<std::unique_ptr<PhyPage>> batch_get(size_t count);

  // Put a physical page back to the pool
  void put(std::unique_ptr<PhyPage> page);

  // Put multiple physical pages back to the pool in one lock
  void batch_put(std::vector<std::unique_ptr<PhyPage>>& pages);

  // Get number of available pages in the pool
  size_t num_available() const;

  // Get total number of pages (available + in use)
  size_t num_total() const { return num_total_pages_; }

  // Get the device
  const torch::Device& device() const { return device_; }

  // Get the zero page (for initializing virtual memory)
  // The returned pointer is owned by PhyPagePool, do not delete it
  PhyPage* get_zero_page();

 private:
  PhyPagePool() = default;
  ~PhyPagePool() = default;
  PhyPagePool(const PhyPagePool&) = delete;
  PhyPagePool& operator=(const PhyPagePool&) = delete;

  bool initialized_ = false;
  torch::Device device_{torch::kCPU};
  size_t num_total_pages_ = 0;

  mutable std::mutex mtx_;
  std::vector<std::unique_ptr<PhyPage>> free_pages_;

  // Zero page for initializing virtual memory (owned by pool)
  std::unique_ptr<PhyPage> zero_page_;
};

}  // namespace xllm
