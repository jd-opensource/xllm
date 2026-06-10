/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "platform/batch_memcpy.h"

#if defined(USE_NPU)
#include "platform/npu/npu_batch_memcpy.h"
#endif

#include <glog/logging.h>

#include "platform/device.h"

namespace xllm {

std::unique_ptr<BatchMemcpy> create_batch_memcpy(const Device& device) {
  CHECK_EQ(Device::type_str(), "npu")
      << "BatchMemcpy currently only supports NPU, but got backend "
      << Device::type_str();
#if defined(USE_NPU)
  auto batch_memcpy = std::make_unique<npu::NPUBatchMemcpy>();
  batch_memcpy->init(device.index());
  return batch_memcpy;
#else
  LOG(FATAL) << "BatchMemcpy requires USE_NPU.";
  return nullptr;
#endif
}

}  // namespace xllm
