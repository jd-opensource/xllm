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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "utils.h"

namespace xllm::kernel::npu {
aclDataType type_info::get_acl_type(const torch::ScalarType& dtype) {
  switch (dtype) {
    case torch::kInt64:
      return ACL_INT64;
    case torch::kInt32:
      return ACL_INT32;
    case torch::kFloat32:
      return ACL_FLOAT;
    case torch::kInt16:
      return ACL_INT16;
    case torch::kFloat16:
      return ACL_FLOAT16;
    case torch::kBFloat16:
      return ACL_BF16;
    case torch::kInt8:
      return ACL_INT8;
    default:
      return ACL_INT32;
  }
}

void create_acltensor(aclTensor** tensor, const torch::Tensor& tensor_data) {
  aclDataType acl_tensor_type =
      type_info::get_acl_type(tensor_data.scalar_type());
  void* deviceData = const_cast<void*>(tensor_data.storage().data());
  c10::SmallVector<int64_t, 8> storageDims;
  storageDims.push_back(tensor_data.storage().nbytes() /
                        tensor_data.itemsize());
  *tensor = aclCreateTensor(tensor_data.sizes().data(),
                            tensor_data.sizes().size(),
                            acl_tensor_type,
                            tensor_data.strides().data(),
                            tensor_data.storage_offset(),
                            aclFormat::ACL_FORMAT_ND,
                            storageDims.data(),
                            storageDims.size(),
                            deviceData);
  if (*tensor == nullptr) {
    LOG(ERROR) << "create_acltensor: failed to create acltensor";
    LOG(FATAL) << "create_acltensor: failed to create acltensor";
  }
}

void check_tensor(const torch::Tensor& t,
                  const std::string& name,
                  const std::string& func_name) {
  if (!t.defined()) {
    LOG(FATAL) << func_name << ": " << name << " is not defined";
  }
  if (t.numel() == 0) {
    LOG(FATAL) << func_name << ": " << name << " is empty";
  }
  if (t.data_ptr() == nullptr) {
    LOG(FATAL) << func_name << ": " << name << " data pointer is null";
  }
}

void check_tensor_shapes_equal(const torch::Tensor& a,
                               const torch::Tensor& b,
                               const std::string& func_name) {
  if (a.sizes() != b.sizes()) {
    LOG(ERROR) << func_name << ": tensor shapes do not match. "
               << "a shape: " << a.sizes() << ", b shape: " << b.sizes();
    LOG(FATAL) << func_name << ": tensor shapes do not match";
  }
}

}  // namespace xllm::kernel::npu
