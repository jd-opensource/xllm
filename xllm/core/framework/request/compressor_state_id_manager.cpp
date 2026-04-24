/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/request/compressor_state_id_manager.h"

#include <glog/logging.h>

namespace xllm {

CompressorStateIdManager& CompressorStateIdManager::get_instance() {
  static CompressorStateIdManager instance;
  return instance;
}

void CompressorStateIdManager::initialize(size_t capacity) {
  get_instance().reset(capacity);
}

bool CompressorStateIdManager::is_initialized() {
  auto& instance = get_instance();
  std::lock_guard<std::mutex> lock(instance.mutex_);
  return instance.initialized_;
}

bool CompressorStateIdManager::try_acquire(int64_t& id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || free_ids_.empty()) {
    return false;
  }
  id = free_ids_.back();
  free_ids_.pop_back();
  return true;
}

void CompressorStateIdManager::release(int64_t id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return;
  }
  if (id < 0 || static_cast<size_t>(id) >= capacity_) {
    LOG(ERROR) << "Invalid compressor state id release: " << id;
    return;
  }
  free_ids_.push_back(id);
}

bool CompressorStateIdManager::empty() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !initialized_ || free_ids_.empty();
}

size_t CompressorStateIdManager::available() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return initialized_ ? free_ids_.size() : 0;
}

size_t CompressorStateIdManager::capacity() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return capacity_;
}

void CompressorStateIdManager::reset(size_t capacity) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    if (capacity_ == capacity) {
      return;
    }
    LOG(WARNING) << "CompressorStateIdManager already initialized, "
                 << "resetting capacity from " << capacity_ << " to "
                 << capacity;
  }
  capacity_ = capacity;
  free_ids_.clear();
  free_ids_.reserve(capacity_);
  for (int64_t id = static_cast<int64_t>(capacity_) - 1; id >= 0; --id) {
    free_ids_.push_back(id);
  }
  initialized_ = true;
}

}  // namespace xllm
