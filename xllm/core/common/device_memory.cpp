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

#include "device_memory.h"

#include <glog/logging.h>
#include <torch/torch.h>

#if (TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 5)
#include <c10/core/CachingDeviceAllocator.h>
#endif

#if defined(USE_NPU)
// #include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#ifdef TORCH_HIGHER_THAN_PTA6
// #include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include <acl/acl.h>
#include <torch_npu/csrc/libs/init_npu.h>
#endif

#if defined(USE_MLU)
#include <torch_mlu/csrc/framework/core/caching_allocator.h>
#endif

namespace xllm {

#if defined(USE_NPU)
struct NPUDeviceMem {
  size_t totalGlobalMem = 0;
  size_t freeMem = 0;
};

NPUDeviceMem getDeviceMemories(int64_t deviceid) {
  NPUDeviceMem memory;
  size_t device_free;
  size_t device_total;
  aclrtGetMemInfo(ACL_HBM_MEM, &device_free, &device_total);
  memory.totalGlobalMem = device_total;
  memory.freeMem = device_free;
  return memory;
}

// returns the total memory in bytes of the device.
int64_t DeviceMemory::total_memory(const torch::Device& device) {
  const c10::DeviceIndex device_index =
      device.has_index() ? device.index() : c10::npu::current_device();
  NPUDeviceMem memory = getDeviceMemories(device_index);
  return static_cast<int64_t>(memory.totalGlobalMem);
}

int64_t DeviceMemory::available_memory(const torch::Device& device) {
  const c10::DeviceIndex device_index =
      device.has_index() ? device.index() : c10::npu::current_device();
  NPUDeviceMem memory = getDeviceMemories(device_index);
  return static_cast<int64_t>(memory.freeMem);
}

#endif

#if defined(USE_MLU)
int64_t DeviceMemory::total_memory(const torch::Device& device) {
  auto info = torch_mlu::MLUCachingAllocator::MemGetInfo(device.index());
  return static_cast<int64_t>(info.second);
}

int64_t DeviceMemory::available_memory(const torch::Device& device) {
  auto info = torch_mlu::MLUCachingAllocator::MemGetInfo(device.index());
  return static_cast<int64_t>(info.first);
}
#endif

}  // namespace xllm
