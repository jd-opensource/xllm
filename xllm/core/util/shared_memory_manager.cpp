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

#include "shared_memory_manager.h"

#include <glog/logging.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstring>

namespace xllm {
std::vector<std::string> SharedMemoryManager::pending_cleanups;
std::mutex SharedMemoryManager::cleanup_mutex;

SharedMemoryManager::SharedMemoryManager(const std::string& name,
                                         size_t size,
                                         bool& is_creator)
    : shm_name_(name), size_(size) {
  // Register cleanup handlers for signals (once per process)
  static std::once_flag flag;
  std::call_once(flag, [] {
    signal(SIGINT, cleanup_handler);
    signal(SIGTERM, cleanup_handler);
    // signal(SIGSEGV, cleanup_handler);
  });

  // First try to create exclusively (O_CREAT | O_EXCL)
  fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
  is_creator = (fd_ != -1);

  // If creation failed, try opening existing
  if (!is_creator) {
    fd_ = shm_open(name.c_str(), O_RDWR, 0666);
    if (fd_ == -1) {
      LOG(FATAL) << "shm_open failed: " << strerror(errno);
    }
  } else {
    // Track created SHM for later cleanup
    std::lock_guard<std::mutex> lock(cleanup_mutex);
    pending_cleanups.push_back(name);
  }

  // Set size for new SHM
  if (is_creator && ftruncate(fd_, size) == -1) {
    close(fd_);
    LOG(FATAL) << "ftruncate failed: " << strerror(errno);
  }

  // Map into process address space
  addr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (addr_ == MAP_FAILED) {
    close(fd_);
    LOG(FATAL) << "mmap failed: " << strerror(errno);
  }

  // Initialize memory to zero.
  std::memset(addr_, 0, size_);
}

SharedMemoryManager::~SharedMemoryManager() {
  // Unmap memory
  LOG(INFO) << "Delete ~SharedMemoryManager";
  if (addr_ != MAP_FAILED) {
    munmap(addr_, size_);
  }

  // Close descriptor
  if (fd_ != -1) {
    close(fd_);
  }

  // Cleanup if we're the creator
  std::lock_guard<std::mutex> lock(cleanup_mutex);
  auto it =
      std::find(pending_cleanups.begin(), pending_cleanups.end(), shm_name_);
  if (it != pending_cleanups.end()) {
    shm_unlink(shm_name_.c_str());
    pending_cleanups.erase(it);
  }
}

void SharedMemoryManager::cleanup_handler(int sig) {
  std::lock_guard<std::mutex> lock(cleanup_mutex);
  LOG(INFO) << "Signal: " << sig << " (" << strsignal(sig) << ")";
  for (const auto& name : pending_cleanups) {
    LOG(INFO) << "SharedMemoryManager cleanup_handler name:" << name;
    shm_unlink(name.c_str());
  }
  exit(sig);
}

}  // namespace xllm