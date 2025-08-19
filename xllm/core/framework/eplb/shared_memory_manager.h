#pragma once

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <torch/torch.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace xllm {

class SharedMemoryManager {
 public:
  explicit SharedMemoryManager(const std::string& name,
                               size_t size,
                               bool& is_creator);

  ~SharedMemoryManager();
  void* allocate(int64_t size, int64_t alignment = alignof(max_align_t));
  void* base_address() const { return addr_; }
  int64_t size() const { return size_; }

 private:
  std::string shm_name_;
  int fd_ = -1;
  void* addr_ = MAP_FAILED;
  int64_t size_ = 0;
  int64_t current_offset_ = 0;
  std::mutex mutex_;

  static void cleanup_handler(int sig);
  static std::vector<std::string> pending_cleanups;
  static std::mutex cleanup_mutex;
};

}  // namespace xllm