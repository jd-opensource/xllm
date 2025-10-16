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
#include <dlpack/dlpack.h>
#include <torch/torch.h>

namespace xllm::kernel::cuda {

DLDataType getDLDataTypeForDLPackv1(const torch::Tensor& t) {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = t.element_size() * 8;
  switch (t.scalar_type()) {
    case torch::ScalarType::UInt1:
    case torch::ScalarType::UInt2:
    case torch::ScalarType::UInt3:
    case torch::ScalarType::UInt4:
    case torch::ScalarType::UInt5:
    case torch::ScalarType::UInt6:
    case torch::ScalarType::UInt7:
    case torch::ScalarType::Byte:
    case torch::ScalarType::UInt16:
    case torch::ScalarType::UInt32:
    case torch::ScalarType::UInt64:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 6
    case torch::ScalarType::Int1:
    case torch::ScalarType::Int2:
    case torch::ScalarType::Int3:
    case torch::ScalarType::Int4:
    case torch::ScalarType::Int5:
    case torch::ScalarType::Int6:
    case torch::ScalarType::Int7:
    case torch::ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
#endif
    case torch::ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case torch::ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case torch::ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case torch::ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case torch::ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case torch::ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case torch::ScalarType::Bool:
      dtype.code = DLDataTypeCode::kDLBool;
      break;
    case torch::ScalarType::ComplexHalf:
    case torch::ScalarType::ComplexFloat:
    case torch::ScalarType::ComplexDouble:
      dtype.code = DLDataTypeCode::kDLComplex;
      break;
    case torch::ScalarType::BFloat16:
      dtype.code = DLDataTypeCode::kDLBfloat;
      break;
    case torch::ScalarType::Float8_e5m2:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2;
      break;
    case torch::ScalarType::Float8_e5m2fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e5m2fnuz;
      break;
    case torch::ScalarType::Float8_e4m3fn:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fn;
      break;
    case torch::ScalarType::Float8_e4m3fnuz:
      dtype.code = DLDataTypeCode::kDLFloat8_e4m3fnuz;
      break;
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 8
    case torch::ScalarType::Float8_e8m0fnu:
      dtype.code = DLDataTypeCode::kDLFloat8_e8m0fnu;
      break;
    case torch::ScalarType::Float4_e2m1fn_x2:
      dtype.code = DLDataTypeCode::kDLFloat4_e2m1fn;
      dtype.lanes = 2;
      dtype.bits = 4;
      break;
#endif
    default:
      TORCH_CHECK(false, "Unsupported scalar type: ");
  }
  return dtype;
}

DLDevice torchDeviceToDLDeviceForDLPackv1(torch::Device device) {
  DLDevice ctx;

  ctx.device_id =
      (device.is_cuda() || device.is_privateuseone())
          ? static_cast<int32_t>(static_cast<unsigned char>(device.index()))
          : 0;

  switch (device.type()) {
    case torch::DeviceType::CPU:
      ctx.device_type = DLDeviceType::kDLCPU;
      break;
    case torch::DeviceType::CUDA:
#ifdef USE_ROCM
      ctx.device_type = DLDeviceType::kDLROCM;
#else
      ctx.device_type = DLDeviceType::kDLCUDA;
#endif
      break;
    case torch::DeviceType::OPENCL:
      ctx.device_type = DLDeviceType::kDLOpenCL;
      break;
    case torch::DeviceType::HIP:
      ctx.device_type = DLDeviceType::kDLROCM;
      break;
    case torch::DeviceType::MAIA:
      ctx.device_type = DLDeviceType::kDLMAIA;
      break;
    case torch::DeviceType::PrivateUse1:
      ctx.device_type = DLDeviceType::kDLExtDev;
      break;
    case torch::DeviceType::MPS:
      ctx.device_type = DLDeviceType::kDLMetal;
      break;
    default:
      TORCH_CHECK(false, "Cannot pack tensors on " + device.str());
  }

  return ctx;
}

template <class T>
struct ATenDLMTensor {
  torch::Tensor handle;
  T tensor{};
};

template <class T>
void deleter(T* arg) {
  delete static_cast<ATenDLMTensor<T>*>(arg->manager_ctx);
}

// Adds version information for DLManagedTensorVersioned.
// This is a no-op for the other types.
template <class T>
void fillVersion(T* tensor) {}

template <>
void fillVersion<DLManagedTensorVersioned>(DLManagedTensorVersioned* tensor) {
  tensor->flags = 0;
  tensor->version.major = DLPACK_MAJOR_VERSION;
  tensor->version.minor = DLPACK_MINOR_VERSION;
}

// This function returns a shared_ptr to memory managed DLpack tensor
// constructed out of ATen tensor
template <class T>
T* toDLPackImpl(const torch::Tensor& src) {
  ATenDLMTensor<T>* atDLMTensor(new ATenDLMTensor<T>);
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.deleter = &deleter<T>;
  atDLMTensor->tensor.dl_tensor.data = src.data_ptr();
  atDLMTensor->tensor.dl_tensor.device =
      torchDeviceToDLDeviceForDLPackv1(src.device());
  atDLMTensor->tensor.dl_tensor.ndim = static_cast<int32_t>(src.dim());
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataTypeForDLPackv1(src);
  atDLMTensor->tensor.dl_tensor.shape =
      const_cast<int64_t*>(src.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides =
      const_cast<int64_t*>(src.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  fillVersion(&atDLMTensor->tensor);
  return &(atDLMTensor->tensor);
}

static torch::Device getATenDeviceForDLPackv1(DLDeviceType type,
                                              c10::DeviceIndex index,
                                              void* data = nullptr) {
  switch (type) {
    case DLDeviceType::kDLCPU:
      return torch::Device(torch::DeviceType::CPU);
#ifndef USE_ROCM
    // if we are compiled under HIP, we cannot do cuda
    case DLDeviceType::kDLCUDA:
      return torch::Device(torch::DeviceType::CUDA, index);
#endif
    case DLDeviceType::kDLOpenCL:
      return torch::Device(torch::DeviceType::OPENCL, index);
    case DLDeviceType::kDLROCM:
#ifdef USE_ROCM
      // this looks funny, we need to return CUDA here to masquerade
      return torch::Device(torch::DeviceType::CUDA, index);
#else
      return torch::Device(torch::DeviceType::HIP, index);
#endif
    case DLDeviceType::kDLMAIA:
      return torch::Device(torch::DeviceType::MAIA, index);
    case DLDeviceType::kDLExtDev:
      return torch::Device(torch::DeviceType::PrivateUse1, index);
    case DLDeviceType::kDLMetal:
      return torch::Device(torch::DeviceType::MPS, index);
    default:
      TORCH_CHECK(false, "Unsupported device_type: ", std::to_string(type));
  }
}

torch::ScalarType toScalarTypeForDLPackv1(const DLDataType& dtype) {
  torch::ScalarType stype = torch::ScalarType::Undefined;
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 8
  if (dtype.code != DLDataTypeCode::kDLFloat4_e2m1fn) {
    TORCH_CHECK(dtype.lanes == 1,
                "ATen does not support lanes != 1 for dtype code",
                std::to_string(dtype.code));
  }
#endif
  switch (dtype.code) {
    case DLDataTypeCode::kDLUInt:
      switch (dtype.bits) {
        case 8:
          stype = torch::ScalarType::Byte;
          break;
        case 16:
          stype = torch::ScalarType::UInt16;
          break;
        case 32:
          stype = torch::ScalarType::UInt32;
          break;
        case 64:
          stype = torch::ScalarType::UInt64;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kUInt bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLInt:
      switch (dtype.bits) {
        case 8:
          stype = torch::ScalarType::Char;
          break;
        case 16:
          stype = torch::ScalarType::Short;
          break;
        case 32:
          stype = torch::ScalarType::Int;
          break;
        case 64:
          stype = torch::ScalarType::Long;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kInt bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat:
      switch (dtype.bits) {
        case 16:
          stype = torch::ScalarType::Half;
          break;
        case 32:
          stype = torch::ScalarType::Float;
          break;
        case 64:
          stype = torch::ScalarType::Double;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBfloat:
      switch (dtype.bits) {
        case 16:
          stype = torch::ScalarType::BFloat16;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLComplex:
      switch (dtype.bits) {
        case 32:
          stype = torch::ScalarType::ComplexHalf;
          break;
        case 64:
          stype = torch::ScalarType::ComplexFloat;
          break;
        case 128:
          stype = torch::ScalarType::ComplexDouble;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kFloat bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLBool:
      switch (dtype.bits) {
        case 8:
          stype = torch::ScalarType::Bool;
          break;
        default:
          TORCH_CHECK(
              false, "Unsupported kDLBool bits ", std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e5m2:
      switch (dtype.bits) {
        case 8:
          stype = torch::ScalarType::Float8_e5m2;
          break;
        default:
          TORCH_CHECK(false,
                      "Unsupported kDLFloat8_e5m2 bits ",
                      std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e5m2fnuz:
      switch (dtype.bits) {
        case 8:
          stype = torch::ScalarType::Float8_e5m2fnuz;
          break;
        default:
          TORCH_CHECK(false,
                      "Unsupported kDLFloat8_e5m2fnuz bits ",
                      std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e4m3fn:
      switch (dtype.bits) {
        case 8:
          stype = torch::ScalarType::Float8_e4m3fn;
          break;
        default:
          TORCH_CHECK(false,
                      "Unsupported kDLFloat8_e4m3fn bits ",
                      std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat8_e4m3fnuz:
      switch (dtype.bits) {
        case 8:
          stype = torch::ScalarType::Float8_e4m3fnuz;
          break;
        default:
          TORCH_CHECK(false,
                      "Unsupported kDLFloat8_e4m3fnuz bits ",
                      std::to_string(dtype.bits));
      }
      break;
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 8
    case DLDataTypeCode::kDLFloat8_e8m0fnu:
      switch (dtype.bits) {
        case 8:
          stype = torch::ScalarType::Float8_e8m0fnu;
          break;
        default:
          TORCH_CHECK(false,
                      "Unsupported kDLFloat8_e8m0fnu bits ",
                      std::to_string(dtype.bits));
      }
      break;
    case DLDataTypeCode::kDLFloat4_e2m1fn:
      switch (dtype.bits) {
        case 4:
          switch (dtype.lanes) {
            case 2:
              stype = torch::ScalarType::Float4_e2m1fn_x2;
              break;
            default:
              TORCH_CHECK(false,
                          "Unsupported kDLFloat4_e2m1fn lanes ",
                          std::to_string(dtype.lanes));
          }
          break;
        default:
          TORCH_CHECK(false,
                      "Unsupported kDLFloat4_e2m1fn bits ",
                      std::to_string(dtype.bits));
      }
      break;
#endif
    default:
      TORCH_CHECK(false, "Unsupported code ", std::to_string(dtype.code));
  }
  return stype;
}

// This function constructs a Tensor from a memory managed DLPack which
// may be represented as either: DLManagedTensor and DLManagedTensorVersioned.
template <class T>
torch::Tensor fromDLPackImpl(T* src, std::function<void(void*)> deleter) {
  if (!deleter) {
    deleter = [src](void* self [[maybe_unused]]) {
      if (src->deleter) {
        src->deleter(src);
      }
    };
  }

  DLTensor& dl_tensor = src->dl_tensor;
  torch::Device device = getATenDeviceForDLPackv1(
      dl_tensor.device.device_type, dl_tensor.device.device_id, dl_tensor.data);
  torch::ScalarType stype = toScalarTypeForDLPackv1(dl_tensor.dtype);

  if (!dl_tensor.strides) {
    return torch::from_blob(dl_tensor.data,
                            torch::IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
                            std::move(deleter),
                            torch::TensorOptions().device(device).dtype(stype));
  }
  return torch::from_blob(dl_tensor.data,
                          torch::IntArrayRef(dl_tensor.shape, dl_tensor.ndim),
                          torch::IntArrayRef(dl_tensor.strides, dl_tensor.ndim),
                          deleter,
                          torch::TensorOptions().device(device).dtype(stype));
}

void toDLPackNonOwningImpl(const torch::Tensor& tensor, DLTensor& out) {
  // Fill in the pre-allocated DLTensor struct with direct pointers
  // This is a non-owning conversion - the caller owns the tensor
  // and must keep it alive for the duration of DLTensor usage
  out.data = tensor.data_ptr();
  out.device = torchDeviceToDLDeviceForDLPackv1(tensor.device());
  out.ndim = static_cast<int32_t>(tensor.dim());
  out.dtype = getDLDataTypeForDLPackv1(tensor);
  // sizes() and strides() return pointers to TensorImpl's stable storage
  // which remains valid as long as the tensor is alive
  out.shape = const_cast<int64_t*>(tensor.sizes().data());
  out.strides = const_cast<int64_t*>(tensor.strides().data());
  out.byte_offset = 0;
}

}  // namespace xllm::kernel::cuda
