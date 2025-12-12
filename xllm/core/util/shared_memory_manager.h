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

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

class SharedMemoryManager {
 public:
  explicit SharedMemoryManager(const std::string& name,
                               size_t size,
                               bool& is_creator);
  virtual ~SharedMemoryManager();
  void* base_address() const { return addr_; }
  int64_t size() const { return size_; }
  std::string name() const { return shm_name_; }

 private:
  std::string shm_name_;
  int fd_ = -1;
  void* addr_ = MAP_FAILED;
  int64_t size_ = 0;
  int64_t current_offset_ = 0;

  static void cleanup_handler(int sig);
  static std::vector<std::string> pending_cleanups;
  static std::mutex cleanup_mutex;
};

}  // namespace xllm