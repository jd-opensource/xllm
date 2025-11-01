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
  std::unique_ptr<Stream> get_stream_from_pool(const int32_t timeout = -1);

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