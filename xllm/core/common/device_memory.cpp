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
// TODO(mlu): implement mlu device memory
#endif

}  // namespace xllm
