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

#include "atb_buffer.h"

#include <acl/acl.h>
#include <atb/types.h>
#include <atb_speed/utils/timer.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>

#include "xllm_kernels/core/include/atb_speed/utils/statistic.h"
#include "xllm_kernels/pytorch/adapter/utils/utils.h"
namespace xllm {
constexpr uint64_t KB_1 = 1024;
constexpr uint64_t MB_1 = 1024 * 1024;
constexpr uint64_t GB_1 = 1024 * 1024 * 1024;
constexpr uint64_t DIM_NUM_2 = 2;
AtbBuffer::AtbBuffer(uint64_t buffer_size, at::Device device)
    : buffer_size_(buffer_size), device_(device) {
  buffer_size_ = buffer_size;
  if (buffer_size_ > 0) {
    at_tensor_ = create_attensor(buffer_size_);
    buffer_ = at_tensor_.data_ptr();
  }
}

AtbBuffer::~AtbBuffer() {}

void* AtbBuffer::get_buffer(uint64_t buffer_size) {
  if (buffer_size <= buffer_size_) {
    return at_tensor_.data_ptr();
  }

  aclrtSynchronizeStream(
      c10_npu::getCurrentNPUStream(device_.index()).stream());

  at_tensor_.reset();
  at_tensor_ = create_attensor(buffer_size);
  buffer_size_ = uint64_t(at_tensor_.numel());
  buffer_ = at_tensor_.data_ptr();

  return at_tensor_.data_ptr();
}

torch::Tensor AtbBuffer::create_attensor(uint64_t buffer_size) const {
  atb::TensorDesc tensorDesc;
  tensorDesc.dtype = ACL_UINT8;
  tensorDesc.format = ACL_FORMAT_ND;

  tensorDesc.shape.dimNum = DIM_NUM_2;
  tensorDesc.shape.dims[0] = KB_1;
  tensorDesc.shape.dims[1] = buffer_size / KB_1 + int(1);

  return create_attensor_from_tensor_desc(tensorDesc);
}

at::Tensor AtbBuffer::create_attensor_from_tensor_desc(
    const atb::TensorDesc& tensorDesc) const {
  static std::map<aclDataType, at::ScalarType> dtypeMap = {
      {ACL_BOOL, at::ScalarType::Bool},
      {ACL_UINT8, at::ScalarType::Byte},
      {ACL_INT8, at::ScalarType::Char},
      {ACL_FLOAT16, at::ScalarType::Half},
      {ACL_FLOAT, at::ScalarType::Float},
      {ACL_INT32, at::ScalarType::Int},
      {ACL_INT64, at::ScalarType::Long},
      {ACL_BF16, at::ScalarType::BFloat16},
  };
  at::TensorOptions options = at::TensorOptions();
  auto it = dtypeMap.find(tensorDesc.dtype);
  if (it != dtypeMap.end()) {
    options = options.dtype(it->second);
  } else {
    throw std::runtime_error("CreateAtTensorFromTensorDesc: not support dtype");
  }

  options =
      options.layout(torch::kStrided).requires_grad(false).device(device_);

  at::Tensor newTensor = at_npu::native::empty_with_format(
      at::IntArrayRef(tensorDesc.shape.dims, tensorDesc.shape.dimNum),
      options,
      tensorDesc.format);

  if (!newTensor.is_contiguous()) {
    newTensor = newTensor.contiguous();
  }

  return newTensor;
}

}  // namespace xllm
