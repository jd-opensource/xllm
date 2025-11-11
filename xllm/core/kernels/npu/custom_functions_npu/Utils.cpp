// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Utils.h"

#include <torch_npu/csrc/core/npu/DeviceUtils.h>

namespace atb {
namespace utils {

ContextManager& ContextManager::GetInstance() {
  static ContextManager instance;
  return instance;
}

ContextManager::ContextManager() : atb_context_(nullptr) {}

ContextManager::~ContextManager() {
  if (atb_context_) {
    auto status = atb::DestroyContext(atb_context_);
    TORCH_CHECK(status == 0, "Destroy context failed!");
    atb_context_ = nullptr;
  }
}

atb::Context* ContextManager::GetContext(aclrtStream stream) {
  std::call_once(create_flag_, [this]() {
    auto status = atb::CreateContext(&atb_context_);
    TORCH_CHECK(status == 0, "Create context failed!");
  });

  atb_context_->SetExecuteStream(stream);
  return atb_context_;
}

atb::Context* GetContext(aclrtStream stream) {
  return ContextManager::GetInstance().GetContext(stream);
}

aclDataType ConvertToAclDataType(const at::ScalarType& data_type) {
  auto acl_dtype =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
  TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED,
              std::string(c10::toString(data_type)) + " has not been supported")
  return acl_dtype;
}

at::Tensor FormatTrans(const at::Tensor& at_tensor) {
  if (torch_npu::utils::is_npu(at_tensor)) {
    return at_npu::native::npu_format_cast(at_tensor, ACL_FORMAT_ND);
  }
  return at_tensor;
}

bool IsBaseFormat(aclFormat& format) {
  return (format == ACL_FORMAT_NCHW) || (format == ACL_FORMAT_ND) ||
         (format == ACL_FORMAT_NHWC) || (format == ACL_FORMAT_NCDHW);
}

aclFormat GetFormatForAtb(const at::Tensor& at_tensor) {
  if (torch_npu::utils::is_npu(at_tensor)) {
    aclFormat format =
        static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
    return IsBaseFormat(format) ? ACL_FORMAT_ND : format;
  }
  return ACL_FORMAT_ND;
}
}  // namespace utils
}  // namespace atb
