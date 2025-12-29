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

#include <atomic>

namespace xllm {

template <class T>
class DoubleBuffer {
 public:
  DoubleBuffer() : index_(0) {}
  virtual ~DoubleBuffer() {}

  T* get_front_value() {
    uint32_t index = front_index();
    return buffers_[index];
  }

  const T* get_front_value() const {
    uint32_t index = front_index();
    return buffers_[index];
  }

  T* get_back_value() {
    uint32_t index = back_index();
    return buffers_[index];
  }

  const T* get_back_value() const {
    uint32_t index = back_index();
    return buffers_[index];
  }

  void set_front_value(T* t) {
    uint32_t index = front_index();
    buffers_[index] = t;
  }

  void set_back_value(T* t) {
    uint32_t index = back_index();
    buffers_[index] = t;
  }

  void swap() {
    uint32_t index = back_index();
    index_.store(index, std::memory_order_release);
  }

 private:
  uint32_t front_index() const {
    return index_.load(std::memory_order_acquire);
  }

  uint32_t back_index() const { return 1 - front_index(); }

  T* buffers_[2] = {nullptr, nullptr};
  std::atomic<uint32_t> index_;
};

enum DoubleBufferIndex {
  kFront = 0,
  kBack,
};

}  // namespace xllm