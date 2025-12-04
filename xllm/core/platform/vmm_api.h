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

#if defined(USE_NPU)
#include "acl/acl.h"
#elif defined(USE_MLU)
#include <cn_api.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <cuda.h>
#endif

namespace xllm {

#if defined(USE_NPU)
using VirPtr = void*;
using PhyMemHandle = aclrtDrvMemHandle;
#elif defined(USE_MLU)
using VirPtr = CNaddr;
using PhyMemHandle = CNmemGenericAllocationHandle;
#elif defined(USE_CUDA) || defined(USE_ILU)
using VirPtr = CUdeviceptr;
using PhyMemHandle = CUmemGenericAllocationHandle;
#endif

namespace vmm {

// create a physical memory handle for a specific device
void create_phy_mem_handle(PhyMemHandle& phy_mem_handle, int32_t device_id);

// create a virtual memory pointer with a specific aligned size
void create_vir_ptr(VirPtr& vir_ptr, size_t aligned_size);

// release a physical memory handle
void release_phy_mem_handle(PhyMemHandle& phy_mem_handle);

// release a virtual memory pointer with a specific aligned size
void release_vir_ptr(VirPtr& vir_ptr, size_t aligned_size);

// map a virtual memory pointer to a physical memory handle
void map(VirPtr& vir_ptr, PhyMemHandle& phy_mem_handle);

// unmap a virtual memory pointer with a specific aligned size
void unmap(VirPtr& vir_ptr, size_t aligned_size);

}  // namespace vmm
}  // namespace xllm