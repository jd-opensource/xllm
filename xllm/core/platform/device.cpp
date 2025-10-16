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

#include "device.h"
#if defined(USE_MLU)
#include <torch_mlu/csrc/framework/core/caching_allocator.h>
#include <torch_mlu/csrc/framework/core/device.h>
#include <torch_mlu/csrc/framework/core/device_utils.h>
#endif

namespace xllm {

Device::Device(torch::Device device) : device_(device) {}

Device::operator torch::Device() const { return unwrap(); }

void Device::set_device() const {
  int ret = 0;
#if defined(USE_NPU)
  ret = c10_npu::SetDevice(index());
#elif defined(USE_MLU)
  torch_mlu::setDevice(index());
#endif

  if (ret != 0) {
    LOG(ERROR) << "set device id: " << index() << " failed, ret:" << ret;
  }
}

const torch::Device& Device::unwrap() const { return device_; }

int32_t Device::index() const { return device_.index(); }

// set device before init device context
void Device::init_device_context() const {
  std::string device_name = type() + ":" + std::to_string(index());
#if defined(USE_NPU)
  torch_npu::init_npu(device_name);
#endif
}

int Device::device_count() {
#if defined(USE_NPU)
  return c10_npu::device_count();
#elif defined(USE_MLU)
  return torch_mlu::device_count();
#endif
}

const std::string Device::type() {
#if defined(USE_NPU)
  return "npu";
#elif defined(USE_MLU)
  return "mlu";
#endif
}

// set device before get device mem
Device::DeviceMem Device::get_device_mem() const {
  DeviceMem device_mem;
  size_t total_memory = 0;
  size_t free_memory = 0;
#if defined(USE_NPU)
  aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
#elif defined(USE_MLU)
  std::tie(free_memory, total_memory) =
      torch_mlu::MLUCachingAllocator::MemGetInfo(index());
#endif
  device_mem.total_memory = static_cast<int64_t>(total_memory);
  device_mem.free_memory = static_cast<int64_t>(free_memory);
  return device_mem;
}

int64_t Device::total_memory() { return get_device_mem().total_memory; }

int64_t Device::free_memory() { return get_device_mem().free_memory; }

int Device::synchronize_default_stream() {
#if defined(USE_NPU)
  return aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(index()).stream());
#elif defined(USE_MLU)
  torch_mlu::getCurrentMLUStream(index()).synchronize();
  return 0;
#endif
}

std::unique_ptr<Stream> Device::get_stream_from_pool() {
  return std::make_unique<Stream>();
}

}  // namespace xllm
