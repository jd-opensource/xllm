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

#include "vmm_api.h"

#include <glog/logging.h>

#include "common/global_flags.h"

namespace xllm {
namespace vmm {

void create_phy_mem_handle(PhyMemHandle& phy_mem_handle, int32_t device_id) {
  int ret = 0;
  // actually, granularity size for physical page is 2MB by default for cuda
  // and npu, but 32MB for mlu
  size_t granularity_size = 0;
#if defined(USE_NPU)
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;  // 2MB
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.reserve = 0;

  // get the recommended granularity size
  ret = aclrtMemGetAllocationGranularity(
      &prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity_size);
  CHECK_EQ(ret, 0) << "Failed to get allocation granularity";

  ret = aclrtMallocPhysical(&phy_mem_handle, granularity_size, &prop, 0);
#elif defined(USE_MLU)
  CNmemAllocationProp prop = {};
  // The memory allocation type requested, which must be
  // CN_MEM_ALLOCATION_TYPE_DEFAULT currently according to cndrv developer
  // guide.
  prop.type =
      CN_MEM_ALLOCATION_TYPE_DEFAULT;  //  same as CU_MEM_ALLOCATION_TYPE_PINNED
                                       //  in CUDA
  prop.location.type = CN_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.requestedHandleTypes = CN_MEM_HANDLE_TYPE_NONE;
  prop.allocFlags.compressionType = CN_MEM_ALLOCATION_COMP_NONE;

  // get the recommended granularity size
  ret = cnMemGetAllocationGranularity(
      &granularity_size, &prop, CN_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  CHECK_EQ(ret, 0) << "Failed to get allocation granularity";

  ret = cnMemCreate(&phy_mem_handle, granularity_size, &prop, 0);

  CNmemAccessDesc accessDesc = {};
  accessDesc.location.type = CN_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device_id;
  accessDesc.accessFlags = CN_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ret = cnMemSetAccess(phy_mem_handle, granularity_size, &accessDesc, 1);
#elif defined(USE_CUDA)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  ret = cuMemCreate(&phy_mem_handle, granularity_size, &prop, 0);

  // get the recommended granularity size
  ret = cuMemGetAllocationGranularity(
      &granularity_size, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  CHECK_EQ(ret, 0) << "Failed to get allocation granularity";

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = device_id;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  ret = cuMemSetAccess(phy_mem_handle, granularity_size, &accessDesc, 1);
#endif
  CHECK_EQ(ret, 0) << "Failed to create physical memory handle";
  FLAGS_phy_page_granularity_size = granularity_size;
  LOG(INFO) << "Granularity size for physical page: " << granularity_size
            << "Bytes";
}

void create_vir_ptr(VirPtr& vir_ptr, size_t aligned_size) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtReserveMemAddress(&vir_ptr, aligned_size, 0, nullptr, 0);
#elif defined(USE_MLU)
  ret = cnMemAddressReserve(&vir_ptr, aligned_size, 0, 0, 0);
#elif defined(USE_CUDA)
  ret = cuMemAddressReserve(&vir_ptr, aligned_size, 0, 0, 0);
#endif
  CHECK_EQ(ret, 0) << "Failed to create virtual memory handle";
}

void release_phy_mem_handle(PhyMemHandle& phy_mem_handle) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtFreePhysical(phy_mem_handle);
#elif defined(USE_MLU)
  ret = cnMemRelease(phy_mem_handle);
#elif defined(USE_CUDA)
  ret = cuMemRelease(phy_mem_handle);
#endif
  CHECK_EQ(ret, 0) << "Failed to release physical memory handle";
}

void release_vir_ptr(VirPtr& vir_ptr, size_t aligned_size) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtReleaseMemAddress(vir_ptr);
#elif defined(USE_MLU)
  ret = cnMemAddressFree(vir_ptr, aligned_size);
#elif defined(USE_CUDA)
  ret = cuMemAddressFree(vir_ptr, aligned_size);
#endif
  CHECK_EQ(ret, 0) << "Failed to release virtual memory handle";
}

void map(VirPtr& vir_ptr, PhyMemHandle& phy_mem_handle) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtMapMem(
      vir_ptr, FLAGS_phy_page_granularity_size, 0, phy_mem_handle, 0);
#elif defined(USE_MLU)
  ret =
      cnMemMap(vir_ptr, FLAGS_phy_page_granularity_size, 0, phy_mem_handle, 0);
#elif defined(USE_CUDA)
  ret =
      cuMemMap(vir_ptr, FLAGS_phy_page_granularity_size, 0, phy_mem_handle, 0);
#endif
  CHECK_EQ(ret, 0) << "Failed to map virtual memory to physical memory";
}

void unmap(VirPtr& vir_ptr, size_t aligned_size) {
  int ret = 0;
#if defined(USE_NPU)
  ret = aclrtUnmapMem(vir_ptr);
#elif defined(USE_MLU)
  ret = cnMemUnmap(vir_ptr, aligned_size);
#elif defined(USE_CUDA)
  ret = cuMemUnmap(vir_ptr, aligned_size);
#endif
  CHECK_EQ(ret, 0) << "Failed to unmap virtual memory from physical memory";
}

}  // namespace vmm
}  // namespace xllm
