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

#include <stdexcept>

#include "ATen/Tensor.h"
#include "cnnl.h"

namespace xllm::mlu {

namespace {
#define CNNL_TYPE_AND_SCALAR_TYPE_WITHOUT_64BIT(_) \
  _(CNNL_DTYPE_FLOAT, at::kFloat)                  \
  _(CNNL_DTYPE_BFLOAT16, at::kBFloat16)            \
  _(CNNL_DTYPE_HALF, at::kHalf)                    \
  _(CNNL_DTYPE_INT32, at::kInt)                    \
  _(CNNL_DTYPE_INT8, at::kChar)                    \
  _(CNNL_DTYPE_UINT8, at::kByte)                   \
  _(CNNL_DTYPE_BOOL, at::kBool)                    \
  _(CNNL_DTYPE_INT16, at::kShort)                  \
  _(CNNL_DTYPE_COMPLEX_HALF, at::kComplexHalf)     \
  _(CNNL_DTYPE_COMPLEX_FLOAT, at::kComplexFloat)   \
  _(CNNL_DTYPE_FLOAT8_E4M3FN, at::kFloat8_e4m3fn)  \
  _(CNNL_DTYPE_FLOAT8_E5M2, at::kFloat8_e5m2)

cnnlDataType_t getCnnlDataType(const at::ScalarType& data_type) {
  switch (data_type) {
#define DEFINE_CASE(cnnl_dtype, scalar_type) \
  case scalar_type:                          \
    return cnnl_dtype;
    CNNL_TYPE_AND_SCALAR_TYPE_WITHOUT_64BIT(DEFINE_CASE)
#undef DEFINE_CASE
    case at::kLong:
      return CNNL_DTYPE_INT32;
    case at::kDouble:
      return CNNL_DTYPE_FLOAT;
    case at::kComplexDouble:
      return CNNL_DTYPE_COMPLEX_FLOAT;
    default:
      std::string msg("getCnnlDataType() not supported for ");
      throw std::runtime_error(msg + c10::toString(data_type));
  }
}

template <typename T>
bool isMlu(const T& tensor) {
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 1)
  return tensor.device().is_privateuseone();
#else
  return tensor.is_mlu();
#endif
}
}  // namespace

using TensorDesc =
    std::unique_ptr<std::remove_pointer_t<cnnlTensorDescriptor_t>,
                    decltype(&cnnlDestroyTensorDescriptor)>;

enum class KernelStatus { KERNEL_STATUS_SUCCESS = 0, KERNEL_STATUS_FAILED };

#define CHECK_SHAPE(x, ...)                                   \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
              #x " must have shape (" #__VA_ARGS__ ")")

#define PAD_UP_DIV(x, y) (((x) + (y) - 1) / (y))

#define CNNL_CHECK_FATAL(expr)                                       \
  if ((expr) != CNNL_STATUS_SUCCESS) {                               \
    std::cerr << __FILE__ << ":" << __LINE__ << ": "                 \
              << " Check failed: " #expr " == CNNL_STATUS_SUCCESS. " \
              << std::endl;                                          \
    throw std::runtime_error("Check failed: " #expr                  \
                             " == CNNL_STATUS_SUCCESS.");            \
  }

#define TMO_KERNEL_CHECK_FATAL(expr)                                      \
  if ((expr) != tmo::KernelStatus::KERNEL_STATUS_SUCCESS) {               \
    std::cerr << __FILE__ << ":" << __LINE__ << ": "                      \
              << " Check failed: " #expr                                  \
                 " == KernelStatus::KERNEL_STATUS_SUCCESS. "              \
              << std::endl;                                               \
    throw std::runtime_error("Check failed: " #expr                       \
                             " == KernelStatus::KERNEL_STATUS_SUCCESS."); \
  }

inline std::vector<TensorDesc> createTensorDescs(
    const std::initializer_list<at::Tensor>& tensors) {
  std::vector<TensorDesc> descs;
  for (size_t i = 0; i < tensors.size(); ++i) {
    descs.emplace_back(TensorDesc{nullptr, cnnlDestroyTensorDescriptor});
    auto tensor = tensors.begin()[i];
    if (!tensor.defined()) {
      continue;
    }
    cnnlTensorDescriptor_t desc;
    CNNL_CHECK_FATAL(cnnlCreateTensorDescriptor(&desc));
    descs[i].reset(desc);
    cnnlDataType_t data_type = getCnnlDataType(tensor.scalar_type());
    if (tensor.strides().size() == 0) {
      CNNL_CHECK_FATAL(cnnlSetTensorDescriptor_v2(descs[i].get(),
                                                  CNNL_LAYOUT_ARRAY,
                                                  data_type,
                                                  tensor.sizes().size(),
                                                  tensor.sizes().data()));
    } else {
      CNNL_CHECK_FATAL(cnnlSetTensorDescriptorEx_v2(descs[i].get(),
                                                    CNNL_LAYOUT_ARRAY,
                                                    data_type,
                                                    tensor.sizes().size(),
                                                    tensor.sizes().data(),
                                                    tensor.strides().data()));
    }
  }
  return descs;
}

inline void* getAtTensorPtr(const c10::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? tensor.value().data_ptr() : nullptr;
}

inline void* getAtTensorPtr(const at::Tensor& tensor) {
  return tensor.defined() ? tensor.data_ptr() : nullptr;
}

enum class TensorAttr { DEVICE, DTYPE, ALL };
enum class TensorDim { LASTDIM, ALL };
struct attr_t {
  int64_t device_id;
  at::ScalarType dtype;
};

inline void checkDevice(int64_t& device_id, const at::Tensor& tensor) {
  auto tensor_device_id = tensor.get_device();
  if (device_id == -1) {
    device_id = tensor_device_id;
    return;
  }
  TORCH_CHECK(tensor_device_id == device_id,
              "Tensor device id is not same, original device_id: ",
              device_id,
              "now device_id is: ",
              tensor_device_id);
}

inline void checkDtype(at::ScalarType& dtype, const at::Tensor& tensor) {
  auto tensor_dtype = tensor.scalar_type();
  if (dtype == at::ScalarType::Undefined) {
    dtype = tensor_dtype;
    return;
  }
  TORCH_CHECK(tensor_dtype == dtype,
              "Tensor dtype is not same. original dtype: ",
              dtype,
              ", now dtype is: ",
              tensor_dtype);
}

template <TensorAttr attr>
inline void checkTensorAttr(attr_t& attr_states, const at::Tensor& tensor) {
  if (attr == TensorAttr::DEVICE) {
    checkDevice(attr_states.device_id, tensor);
  } else if (attr == TensorAttr::DTYPE) {
    checkDtype(attr_states.dtype, tensor);
  } else if (attr == TensorAttr::ALL) {
    checkDevice(attr_states.device_id, tensor);
    checkDtype(attr_states.dtype, tensor);
  }
}

template <
    TensorAttr attr,
    typename T,
    typename = typename std::enable_if<
        std::is_same<typename std::decay<T>::type, at::Tensor>::value>::type>
void checkTensorSameWithSpecificAttr(attr_t& attr_states,
                                     const c10::optional<T>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) return;
  auto temp_tensor = tensor.value();
  TORCH_CHECK(isMlu(temp_tensor), "Only support mlu tensor.");
  checkTensorAttr<attr>(attr_states, temp_tensor);
}

template <
    TensorAttr attr,
    typename T,
    typename = typename std::enable_if<
        std::is_same<typename std::decay<T>::type, at::Tensor>::value>::type>
void checkTensorSameWithSpecificAttr(attr_t& attr_states, const T& tensor) {
  if (!tensor.defined()) return;
  TORCH_CHECK(isMlu(tensor), "Only support mlu tensor.");
  checkTensorAttr<attr>(attr_states, tensor);
}

template <TensorAttr attr, typename T, typename... Args>
void checkTensorSameWithSpecificAttr(attr_t& attr_states,
                                     const T& tensor,
                                     Args&&... args) {
  checkTensorSameWithSpecificAttr<attr>(attr_states, tensor);
  checkTensorSameWithSpecificAttr<attr>(attr_states,
                                        std::forward<Args>(args)...);
}

template <TensorAttr attr = TensorAttr::ALL, typename... Args>
void checkTensorSameAttr(Args&&... args) {
  attr_t attr_states = {-1, at::ScalarType::Undefined};
  checkTensorSameWithSpecificAttr<attr>(attr_states,
                                        std::forward<Args>(args)...);
}

template <
    TensorDim dim,
    typename T,
    typename = typename std::enable_if<
        std::is_same<typename std::decay<T>::type, at::Tensor>::value>::type>
void checkTensorContiguousImpl(const std::string& err_msg, const T& tensor) {
  if (!tensor.defined()) return;
  if (dim == TensorDim::ALL) {
    TORCH_CHECK(tensor.is_contiguous(), err_msg);
  } else if (dim == TensorDim::LASTDIM) {
    TORCH_CHECK(tensor.stride(-1) == 1, err_msg);
  }
}

template <
    TensorDim dim,
    typename T,
    typename = typename std::enable_if<
        std::is_same<typename std::decay<T>::type, at::Tensor>::value>::type>
void checkTensorContiguousImpl(const std::string& err_msg,
                               const c10::optional<T>& tensor) {
  if (!tensor.has_value() || !tensor->defined()) return;
  if (dim == TensorDim::ALL) {
    TORCH_CHECK(!tensor.has_value() || tensor.value().is_contiguous(), err_msg);
  } else if (dim == TensorDim::LASTDIM) {
    TORCH_CHECK(!tensor.has_value() || tensor.value().stride(-1) == 1, err_msg);
  }
}

template <TensorDim dim, typename T, typename... Args>
void checkTensorContiguousImpl(const std::string& err_msg,
                               const T& tensor,
                               Args&&... args) {
  checkTensorContiguousImpl<dim>(err_msg, tensor);
  checkTensorContiguousImpl<dim>(err_msg, std::forward<Args>(args)...);
}

template <TensorDim dim = TensorDim::ALL, typename... Args>
void checkTensorContiguous(const std::string& err_msg, Args&&... args) {
  checkTensorContiguousImpl<dim>(err_msg, std::forward<Args>(args)...);
}

}  // namespace xllm::mlu
