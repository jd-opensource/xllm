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

#ifndef OPPLUGIN_UTILS_ATB_UTILS_H
#define OPPLUGIN_UTILS_ATB_UTILS_H

#include <ATen/ATen.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include "atb/atb_infer.h"

namespace atb {
namespace utils {

class ContextManager {
 public:
  static ContextManager& GetInstance();
  atb::Context* GetContext(aclrtStream stream);
  ~ContextManager();

  ContextManager(const ContextManager&) = delete;
  ContextManager& operator=(const ContextManager&) = delete;

 private:
  ContextManager();
  std::once_flag create_flag_;
  atb::Context* atb_context_;
};

atb::Context* GetContext(aclrtStream stream);

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)  \
  _(at::ScalarType::Byte, ACL_UINT8)                 \
  _(at::ScalarType::Char, ACL_INT8)                  \
  _(at::ScalarType::Short, ACL_INT16)                \
  _(at::ScalarType::Int, ACL_INT32)                  \
  _(at::ScalarType::Long, ACL_INT64)                 \
  _(at::ScalarType::Half, ACL_FLOAT16)               \
  _(at::ScalarType::Float, ACL_FLOAT)                \
  _(at::ScalarType::Double, ACL_DOUBLE)              \
  _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)      \
  _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)     \
  _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)   \
  _(at::ScalarType::Bool, ACL_BOOL)                  \
  _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)         \
  _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::BFloat16, ACL_BF16)              \
  _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)      \
  _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)      \
  _(at::ScalarType::Bits1x8, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::Bits2x4, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::Bits4x2, ACL_DT_UNDEFINED)       \
  _(at::ScalarType::Bits8, ACL_DT_UNDEFINED)         \
  _(at::ScalarType::Bits16, ACL_DT_UNDEFINED)        \
  _(at::ScalarType::Float8_e5m2, ACL_DT_UNDEFINED)   \
  _(at::ScalarType::Float8_e4m3fn, ACL_DT_UNDEFINED) \
  _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)     \
  _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable
    [static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
        AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

aclDataType ConvertToAclDataType(const at::ScalarType& data_type);
at::Tensor FormatTrans(const at::Tensor& at_tensor);
aclFormat GetFormatForAtb(const at::Tensor& at_tensor);

template <typename MapType>
inline int get_op_mode(const MapType& mode_map,
                       c10::optional<c10::string_view> mode_opt,
                       c10::string_view default_mode,
                       const char* mode_name) {
  c10::string_view mode_str = mode_opt.value_or(default_mode);
  auto it = mode_map.find(mode_str);
  TORCH_CHECK(it != mode_map.end(),
              "Unsupported ",
              mode_name,
              " value: '",
              mode_str,
              "'");
  return it->second;
}
}  // namespace utils
}  // namespace atb

#endif
