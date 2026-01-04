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

#include "phy_page_pool.h"

#include <glog/logging.h>

namespace xllm {

void PhyPagePool::init(const torch::Device& device, size_t num_pages) {
  std::lock_guard<std::mutex> lock(mtx_);

  if (initialized_) {
    LOG(WARNING) << "PhyPagePool already initialized, ignoring re-init";
    return;
  }

  device_ = device;
  num_total_pages_ = num_pages;

  LOG(INFO) << "PhyPagePool: pre-allocating " << num_pages
            << " physical pages on device " << device;

  // Pre-allocate zero page first (used by all XTensors for initialization)
  zero_page_ = std::make_unique<PhyPage>(device_);

  // Pre-allocate all physical pages for data
  free_pages_.reserve(num_pages);
  for (size_t i = 0; i < num_pages; ++i) {
    free_pages_.push_back(std::make_unique<PhyPage>(device_));
  }

  initialized_ = true;

  LOG(INFO) << "PhyPagePool: successfully pre-allocated " << num_pages
            << " physical pages + 1 zero page";
}

std::unique_ptr<PhyPage> PhyPagePool::get() {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (free_pages_.empty()) {
    LOG(WARNING) << "PhyPagePool: no free pages available";
    return nullptr;
  }

  // LIFO: pop from back (O(1), cache-friendly)
  auto page = std::move(free_pages_.back());
  free_pages_.pop_back();
  return page;
}

std::vector<std::unique_ptr<PhyPage>> PhyPagePool::batch_get(size_t count) {
  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  if (count == 0) {
    return {};
  }

  if (free_pages_.size() < count) {
    LOG(WARNING) << "PhyPagePool: not enough free pages, requested " << count
                 << ", available " << free_pages_.size();
    return {};
  }

  std::vector<std::unique_ptr<PhyPage>> result;
  result.reserve(count);

  // LIFO: pop from back (O(1), cache-friendly)
  for (size_t i = 0; i < count; ++i) {
    result.push_back(std::move(free_pages_.back()));
    free_pages_.pop_back();
  }

  return result;
}

void PhyPagePool::put(std::unique_ptr<PhyPage> page) {
  if (page == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  // Verify the page belongs to this pool (same device)
  CHECK(page->device() == device_) << "Page device mismatch: expected "
                                   << device_ << ", got " << page->device();

  free_pages_.push_back(std::move(page));
}

void PhyPagePool::batch_put(std::vector<std::unique_ptr<PhyPage>>& pages) {
  if (pages.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(mtx_);

  CHECK(initialized_) << "PhyPagePool not initialized";

  for (auto& page : pages) {
    if (page == nullptr) {
      continue;
    }
    // Verify the page belongs to this pool (same device)
    CHECK(page->device() == device_) << "Page device mismatch: expected "
                                     << device_ << ", got " << page->device();

    free_pages_.push_back(std::move(page));
  }
  pages.clear();
}

size_t PhyPagePool::num_available() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return free_pages_.size();
}

PhyPage* PhyPagePool::get_zero_page() {
  std::lock_guard<std::mutex> lock(mtx_);
  CHECK(initialized_) << "PhyPagePool not initialized";
  CHECK(zero_page_) << "Zero page not created";
  return zero_page_.get();
}

}  // namespace xllm
