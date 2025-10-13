#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "stream.h"

namespace xllm {

class Device {
 public:
  explicit Device(torch::Device device);
  ~Device() = default;
  operator torch::Device() const;

  void set_device() const;

  const torch::Device& unwrap() const;
  int32_t index() const;

  void init_device_context() const;

  static int device_count();
  static const std::string type();

  int64_t total_memory();
  int64_t free_memory();

  int synchronize_default_stream();
  std::unique_ptr<Stream> get_stream_from_pool();

 private:
  struct DeviceMem {
    int64_t total_memory;
    int64_t free_memory;
  };

  DeviceMem get_device_mem() const;

 private:
  torch::Device device_;
};

}  // namespace xllm