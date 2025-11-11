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

#include "AtbCommon.h"

namespace atb {
atb::Tensor AtTensor2AtbTensor(const at::Tensor at_tensor) {
  static std::map<at::ScalarType, aclDataType> dtype_map = {
      {at::ScalarType::Bool, ACL_BOOL},
      {at::ScalarType::Byte, ACL_UINT8},
      {at::ScalarType::Char, ACL_INT8},
      {at::ScalarType::Half, ACL_FLOAT16},
      {at::ScalarType::Float, ACL_FLOAT},
      {at::ScalarType::Int, ACL_INT32},
      {at::ScalarType::Long, ACL_INT64},
      {at::ScalarType::BFloat16, ACL_BF16},
      {at::ScalarType::Double, ACL_DOUBLE},
      {at::ScalarType::Short, ACL_INT16},
      {at::ScalarType::ComplexHalf, ACL_COMPLEX32},
      {at::ScalarType::ComplexFloat, ACL_COMPLEX64},
      {at::ScalarType::ComplexDouble, ACL_COMPLEX128},
  };

  TORCH_CHECK(at_tensor.is_contiguous(), "at_tensor is not contiguous");
  atb::Tensor tensor;
  tensor.desc.format = atb::utils::GetFormatForAtb(at_tensor);
  if (at_tensor.device().type() == at::kCPU) {
    tensor.hostData = at_tensor.data_ptr();
  } else {
    tensor.deviceData = at_tensor.data_ptr();
  }

  tensor.desc.shape.dimNum = at_tensor.sizes().size();
  for (uint64_t i = 0; i < at_tensor.sizes().size(); i++) {
    tensor.desc.shape.dims[i] = at_tensor.sizes()[i];
  }

  auto dtype_iterator = dtype_map.find(at_tensor.scalar_type());
  TORCH_CHECK(dtype_iterator != dtype_map.end(),
              "not support dtype: ",
              at_tensor.scalar_type());
  tensor.desc.dtype = dtype_iterator->second;

  tensor.dataSize = atb::Utils::GetTensorSize(tensor);

  return tensor;
}

void RunAtbCmdV1(atb::Operation* op,
                 const ParamSetter& paramsetter,
                 const std::string& name) {
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
  auto context_ptr = atb::utils::GetContext(stream);
  atb::VariantPack variant_pack = paramsetter.variant_pack_;
  uint64_t workspace_size = OperationSetup(variant_pack, op, context_ptr);
  at::Tensor workspace_tensor;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    at::TensorOptions options = at::TensorOptions(c10::DeviceType::PrivateUse1);
    workspace_tensor = at::empty({workspace_size}, options.dtype(at::kByte));
    workspace_ptr = const_cast<void*>(workspace_tensor.storage().data());
  }
  const c10::SmallVector<at::Tensor, N>& cpu_tensors =
      paramsetter.tensor_maintainer_.cpu_tensors;
  auto acl_call = [variant_pack,
                   workspace_ptr,
                   workspace_size,
                   context_ptr,
                   op,
                   cpu_tensors]() -> int {
    auto st = op->Execute(
        variant_pack, (uint8_t*)workspace_ptr, workspace_size, context_ptr);
    DestroyOperation(op);
    return st;
  };
  at_npu::native::OpCommand::RunOpApiV2(name, acl_call);
}

void RunAtbCmdV2(atb::Operation* op,
                 const ParamSetter& paramsetter,
                 const std::string& name) {
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);
  atb::VariantPack variant_pack = paramsetter.variant_pack_;
  const c10::SmallVector<at::Tensor, N>& cpu_tensors =
      paramsetter.tensor_maintainer_.cpu_tensors;
  auto acl_call = [op, variant_pack, stream, cpu_tensors]() -> int {
    auto context_ptr = atb::utils::GetContext(stream);
    uint64_t workspace_size = OperationSetup(variant_pack, op, context_ptr);
    at::Tensor workspace_tensor;
    void* workspace_ptr = nullptr;
    if (workspace_size != 0) {
      workspace_tensor =
          at_npu::native::allocate_workspace(workspace_size, stream);
      workspace_ptr = const_cast<void*>(workspace_tensor.storage().data());
    }
    auto st = op->Execute(
        variant_pack, (uint8_t*)workspace_ptr, workspace_size, context_ptr);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2(name, acl_call);
}

void RunAtbCmd(atb::Operation* op,
               const ParamSetter& paramsetter,
               const std::string& name) {
  const auto is_capturing =
      static_cast<int>(c10_npu::currentStreamCaptureStatusMayInitCtx());
  if (is_capturing) {
    RunAtbCmdV1(op, paramsetter, name);
  } else {
    RunAtbCmdV2(op, paramsetter, name);
  }
}

ParamSetter& ParamSetter::Input(const at::Tensor& tensor,
                                const bool& format_trans) {
  if (!tensor.defined()) {
    variant_pack_.inTensors.push_back(atb::Tensor());
    return *this;
  }
  at::Tensor new_tensor = tensor.contiguous();
  if (format_trans) {
    new_tensor = atb::utils::FormatTrans(new_tensor);
  }
  atb::Tensor atb_tensor;
  if (new_tensor.device().type() == at::kCPU) {
    auto tensor_clone = new_tensor.clone();
    atb_tensor = AtTensor2AtbTensor(tensor_clone);
    tensor_maintainer_.cpu_tensors.emplace_back(std::move(tensor_clone));
  } else {
    atb_tensor = AtTensor2AtbTensor(new_tensor);
    tensor_maintainer_.contiguous_tensors.emplace_back(std::move(new_tensor));
  }
  variant_pack_.inTensors.push_back(atb_tensor);
  return *this;
}

ParamSetter& ParamSetter::Input(const c10::optional<at::Tensor>& tensor,
                                const bool& format_trans) {
  if (!tensor.has_value()) {
    variant_pack_.inTensors.push_back(atb::Tensor());
    return *this;
  }
  return Input(tensor.value(), format_trans);
}

ParamSetter& ParamSetter::Output(at::Tensor& output) {
  auto atb_tensor = AtTensor2AtbTensor(output);
  variant_pack_.outTensors.push_back(atb_tensor);
  return *this;
}

uint64_t OperationSetup(atb::VariantPack variant_pack,
                        atb::Operation* operation,
                        atb::Context* context_ptr) {
  uint64_t workspace_size = 0;
  atb::Status status =
      operation->Setup(variant_pack, workspace_size, context_ptr);
  TORCH_CHECK(status == 0, operation->GetName(), " setup failed!");
  return workspace_size;
}

}  // namespace atb