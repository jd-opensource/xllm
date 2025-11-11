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

#ifndef OPPLUGIN_UTILS_ATB_COMMON_H
#define OPPLUGIN_UTILS_ATB_COMMON_H
#include <dlfcn.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>

#include "./OperationCreate.h"
#include "Utils.h"
#include "atb/atb_infer.h"

namespace atb {

using aclTensor = struct aclTensor;
constexpr int64_t MAX_DIM_NUM = 5;
// small vector max size
const int N = 32;

using _aclCreateTensor = aclTensor* (*)(const int64_t* view_dims,
                                        uint64_t view_dims_num,
                                        aclDataType data_type,
                                        const int64_t* stride,
                                        int64_t offset,
                                        aclFormat format,
                                        const int64_t* storage_dims,
                                        uint64_t storage_dims_num,
                                        void* tensor_data);
using _aclDestroyTensor = int (*)(const aclTensor*);

using AtbApiFunc = int (*)(void*, uint64_t, atb::Operation*, atb::Context*);

#define GET_OP_API_FUNC(apiName) \
  reinterpret_cast<_##apiName>(GetApiFuncAddr(#apiName))

inline const char* GetAtbApiLibName(void) { return "libatb.so"; }

inline const char* GetOpApiLibName(void) { return "libopapi.so"; }

inline void* GetApiLibHandler(const char* libName) {
  auto handler = dlopen(libName, RTLD_LAZY);
  if (handler == nullptr) {
    ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
  }
  return handler;
}

inline void* GetApiFuncAddrInLib(void* handler,
                                 const char* libName,
                                 const char* apiName) {
  auto funcAddr = dlsym(handler, apiName);
  if (funcAddr == nullptr) {
    ASCEND_LOGW(
        "dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
  }
  return funcAddr;
}

inline void* GetApiFuncAddr(const char* apiName) {
  static auto atbApiHandler = GetApiLibHandler(GetAtbApiLibName());
  if (atbApiHandler != nullptr) {
    auto funcAddr =
        GetApiFuncAddrInLib(atbApiHandler, GetAtbApiLibName(), apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
  }
  static auto opApiHandler = GetApiLibHandler(GetOpApiLibName());
  if (opApiHandler != nullptr) {
    auto funcAddr =
        GetApiFuncAddrInLib(opApiHandler, GetOpApiLibName(), apiName);
    if (funcAddr != nullptr) {
      return funcAddr;
    }
    TORCH_CHECK(false, "GetApiFuncAddr not found ", apiName);
  }
}

struct TensorMaintainer {
  c10::SmallVector<at::Tensor, N>
      contiguous_tensors;  // npu tensor's life should maintain when
                           // uncontiguous to contiguous.
  c10::SmallVector<at::Tensor, N>
      cpu_tensors;  // cpu tensor's life should maintain in taskqueue.
};

inline aclTensor* ConvertType(TensorMaintainer& maintainer,
                              const at::Tensor& tensor) {
  static const auto aclCreateTensor =
      reinterpret_cast<_aclCreateTensor>(GetApiFuncAddr("aclCreateTensor"));
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (!tensor.defined()) {
    return nullptr;
  }
  at::Tensor at_tensor = tensor.contiguous();
  aclFormat format = atb::utils::GetFormatForAtb(at_tensor);

  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_data_type =
      atb::utils::ConvertToAclDataType(scalar_data_type);
  c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
  // if acl_data_type is ACL_STRING, storageDims is empty.
  if (acl_data_type != ACL_STRING) {
    TORCH_CHECK(at_tensor.itemsize() > 0,
                "the itemsize of tensor must be greater than 0.");
    storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
  }

  const auto dimNum = at_tensor.sizes().size();
  auto acl_tensor =
      aclCreateTensor(at_tensor.sizes().data(),
                      at_tensor.sizes().size(),
                      acl_data_type,
                      at_tensor.strides().data(),
                      at_tensor.storage_offset(),
                      format,
                      storageDims.data(),
                      storageDims.size(),
                      const_cast<void*>(at_tensor.storage().data()));
  if (at_tensor.device().type() == at::kCPU) {
    maintainer.cpu_tensors.emplace_back(std::move(at_tensor));
  } else {
    maintainer.contiguous_tensors.emplace_back(std::move(at_tensor));
  }
  return acl_tensor;
}

inline aclTensor* ConvertType(TensorMaintainer& maintainer,
                              const c10::optional<at::Tensor>& opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return ConvertType(maintainer, opt_tensor.value());
  }

  return nullptr;
}

template <typename T>
T ConvertType(TensorMaintainer& maintainer, T value) {
  return value;
}

template <typename... Ts>
constexpr auto ConvertTypes(TensorMaintainer& maintainer, Ts&... args) {
  return std::make_tuple(ConvertType(maintainer, args)...);
}

struct TensorStruct {
  void* data_ptr = nullptr;      // at_tensor.storage().data()
  at::ScalarType scalar_type;    // at_tensor.scalar_type()
  size_t nbytes;                 // at_tensor.storage().nbytes()
  size_t itemsize;               // at_tensor.itemsize()
  int64_t storage_offset;        // at_tensor.storage_offset()
  std::vector<int64_t> sizes;    // at_tensor.sizes()
  std::vector<int64_t> strides;  // at_tensor.strides()
  aclFormat format;              // at_tensor format

  TensorStruct(void* data_ptr_,
               at::ScalarType scalar_type_,
               size_t nbytes_,
               size_t itemsize_,
               int64_t storage_offset_,
               at::IntArrayRef sizes_,
               at::IntArrayRef strides_,
               aclFormat format_)
      : data_ptr(data_ptr_),
        scalar_type(scalar_type_),
        nbytes(nbytes_),
        itemsize(itemsize_),
        storage_offset(storage_offset_),
        sizes(sizes_.vec()),
        strides(strides_.vec()),
        format(format_) {}
};
using TensorStructPtr = std::shared_ptr<TensorStruct>;

inline TensorStructPtr CopyTypeV2(TensorMaintainer& maintainer,
                                  const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return nullptr;
  }
  at::Tensor at_tensor = tensor.contiguous();
  aclFormat format = atb::utils::GetFormatForAtb(at_tensor);
  std::shared_ptr<TensorStruct> tensor_structptr =
      std::make_shared<TensorStruct>(
          const_cast<void*>(at_tensor.storage().data()),
          at_tensor.scalar_type(),
          at_tensor.storage().nbytes(),
          at_tensor.itemsize(),
          at_tensor.storage_offset(),
          at_tensor.sizes(),
          at_tensor.strides(),
          format);
  if (at_tensor.device().type() == at::kCPU) {
    maintainer.cpu_tensors.emplace_back(std::move(at_tensor));
  } else {
    maintainer.contiguous_tensors.emplace_back(std::move(at_tensor));
  }
  return tensor_structptr;
}

inline TensorStructPtr CopyTypeV2(TensorMaintainer& maintainer,
                                  const c10::optional<at::Tensor>& opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return CopyTypeV2(maintainer, opt_tensor.value());
  }

  return nullptr;
}

template <typename T>
T CopyTypeV2(TensorMaintainer& maintainer, T value) {
  return value;
}

inline aclTensor* ConvertTypeV2(TensorStructPtr at_tensor) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (at_tensor == nullptr) {
    return nullptr;
  }
  at::ScalarType scalar_data_type = (*at_tensor).scalar_type;
  aclDataType acl_data_type =
      atb::utils::ConvertToAclDataType(scalar_data_type);
  c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
  // if acl_data_type is ACL_STRING, storageDims is empty.
  if (acl_data_type != ACL_STRING) {
    TORCH_CHECK((*at_tensor).itemsize > 0,
                "the itemsize of tensor must be greater than 0.");
    storageDims.push_back((*at_tensor).nbytes / (*at_tensor).itemsize);
  }

  const auto dimNum = (*at_tensor).sizes.size();

  auto acl_tensor = aclCreateTensor((*at_tensor).sizes.data(),
                                    (*at_tensor).sizes.size(),
                                    acl_data_type,
                                    (*at_tensor).strides.data(),
                                    (*at_tensor).storage_offset,
                                    (*at_tensor).format,
                                    storageDims.data(),
                                    storageDims.size(),
                                    (*at_tensor).data_ptr);
  return acl_tensor;
}

template <typename T>
T ConvertTypeV2(T value) {
  return value;
}

template <typename Tuple, std::size_t... I>
auto convert_types_impl_v2(const Tuple& t, std::index_sequence<I...>) {
  return std::make_tuple(ConvertTypeV2(std::get<I>(t))...);
}

template <typename... Ts>
constexpr auto ConvertTypesV2(const std::tuple<Ts...>& args,
                              uint64_t* workspace_size_addr,
                              atb::Operation** op_addr,
                              atb::Context* context_ptr) {
  auto convert_args =
      convert_types_impl_v2(args, std::make_index_sequence<sizeof...(Ts)>{});
  auto appends = std::make_tuple(workspace_size_addr, op_addr, context_ptr);
  return std::tuple_cat(convert_args, appends);
}

template <typename... Ts>
constexpr auto CopyTypesV2(TensorMaintainer& maintainer, Ts&... args) {
  return std::make_tuple(CopyTypeV2(maintainer, args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params,
                        void* opApiAddr,
                        std::index_sequence<I...>) {
  using OpApiFunc =
      int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(
      params, opApiAddr, std::make_index_sequence<size>{});
}

inline void Release(atb::Context* context) {}

inline void Release(aclTensor* p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

template <typename T>
void Release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple& t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

#define EXEC_ATB_CMD_V1(atb_api, ...)                                        \
  do {                                                                       \
    static const auto getWorkspaceSizeFuncAddr =                             \
        GetApiFuncAddr(#atb_api "GetWorkspaceSize");                         \
    static const auto atbApiFuncAddr = GetApiFuncAddr(#atb_api);             \
    TORCH_CHECK(                                                             \
        getWorkspaceSizeFuncAddr != nullptr && atbApiFuncAddr != nullptr,    \
        #atb_api,                                                            \
        " or ",                                                              \
        #atb_api "GetWorkspaceSize",                                         \
        " not in ",                                                          \
        GetAtbApiLibName(),                                                  \
        ", or ",                                                             \
        GetAtbApiLibName(),                                                  \
        "not found.");                                                       \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);          \
    auto context_ptr = atb::utils::GetContext(acl_stream);                   \
    uint64_t workspace_size = 0;                                             \
    uint64_t* workspace_size_addr = &workspace_size;                         \
    atb::Operation* op = nullptr;                                            \
    atb::Operation** op_addr = &op;                                          \
    TensorMaintainer tensor_maintainer;                                      \
    auto converted_params = ConvertTypes(tensor_maintainer,                  \
                                         __VA_ARGS__,                        \
                                         workspace_size_addr,                \
                                         op_addr,                            \
                                         context_ptr);                       \
    static auto getWorkspaceSizeFunc =                                       \
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);      \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);    \
    TORCH_CHECK(workspace_status == 0, "call " #atb_api " failed, detail:"); \
    void* workspace_addr = nullptr;                                          \
    at::Tensor workspace_tensor;                                             \
    if (workspace_size != 0) {                                               \
      at::TensorOptions options =                                            \
          at::TensorOptions(c10::DeviceType::PrivateUse1);                   \
      workspace_tensor =                                                     \
          at::empty({workspace_size}, options.dtype(at::kByte));             \
      workspace_addr = const_cast<void*>(workspace_tensor.storage().data()); \
    }                                                                        \
    const c10::SmallVector<at::Tensor, N>& cpu_tensors =                     \
        tensor_maintainer.cpu_tensors;                                       \
    auto atb_call = [converted_params,                                       \
                     workspace_addr,                                         \
                     workspace_size,                                         \
                     context_ptr,                                            \
                     op,                                                     \
                     cpu_tensors]() -> int {                                 \
      AtbApiFunc atbApiFunc = reinterpret_cast<AtbApiFunc>(atbApiFuncAddr);  \
      auto api_ret =                                                         \
          atbApiFunc(workspace_addr, workspace_size, op, context_ptr);       \
      TORCH_CHECK(api_ret == 0, "call " #atb_api " failed, detail:");        \
      DestroyOperation(op);                                                  \
      ReleaseConvertTypes(converted_params);                                 \
      return api_ret;                                                        \
    };                                                                       \
    at_npu::native::OpCommand::RunOpApiV2(#atb_api, atb_call);               \
  } while (false)

#define EXEC_ATB_CMD_V2(atb_api, ...)                                          \
  do {                                                                         \
    static const auto getWorkspaceSizeFuncAddr =                               \
        GetApiFuncAddr(#atb_api "GetWorkspaceSize");                           \
    static const auto AtbApiFuncAddr = GetApiFuncAddr(#atb_api);               \
    TORCH_CHECK(                                                               \
        getWorkspaceSizeFuncAddr != nullptr && AtbApiFuncAddr != nullptr,      \
        #atb_api,                                                              \
        " or ",                                                                \
        #atb_api "GetWorkspaceSize",                                           \
        " not in ",                                                            \
        GetAtbApiLibName(),                                                    \
        ", or ",                                                               \
        GetAtbApiLibName(),                                                    \
        "not found.");                                                         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);            \
    TensorMaintainer tensor_maintainer;                                        \
    auto copied_params = CopyTypesV2(tensor_maintainer, __VA_ARGS__);          \
    auto hash_id = computeHash(std::string(#atb_api), __VA_ARGS__);            \
    const c10::SmallVector<at::Tensor, N>& cpu_tensors =                       \
        tensor_maintainer.cpu_tensors;                                         \
    auto atb_call =                                                            \
        [copied_params, acl_stream, hash_id, cpu_tensors]() -> int {           \
      auto context_ptr = atb::utils::GetContext(acl_stream);                   \
      uint64_t workspace_size = 0;                                             \
      uint64_t* workspace_size_addr = &workspace_size;                         \
      OpParamCache<uint64_t>& opParamCache =                                   \
          OpParamCache<uint64_t>::getInstance();                               \
      atb::Operation* op = opParamCache.getOperation(hash_id);                 \
      atb::Operation** op_addr = &op;                                          \
      int api_ret = 0;                                                         \
      auto converted_params = ConvertTypesV2(                                  \
          copied_params, workspace_size_addr, op_addr, context_ptr);           \
      auto getWorkspaceSizeFunc =                                              \
          ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);      \
      auto workspace_status = call(getWorkspaceSizeFunc, converted_params);    \
      opParamCache.saveOperation(hash_id, op);                                 \
      TORCH_CHECK(workspace_status == 0,                                       \
                  "call " #atb_api "GetWorkspaceSize failed");                 \
      void* workspace_addr = nullptr;                                          \
      at::Tensor workspace_tensor;                                             \
      if (workspace_size != 0) {                                               \
        workspace_tensor =                                                     \
            at_npu::native::allocate_workspace(workspace_size, acl_stream);    \
        workspace_addr = const_cast<void*>(workspace_tensor.storage().data()); \
      }                                                                        \
      AtbApiFunc atbApiFunc = reinterpret_cast<AtbApiFunc>(AtbApiFuncAddr);    \
      api_ret = atbApiFunc(workspace_addr, workspace_size, op, context_ptr);   \
      TORCH_CHECK(api_ret == 0, "call " #atb_api " failed");                   \
      ReleaseConvertTypes(converted_params);                                   \
      return api_ret;                                                          \
    };                                                                         \
    at_npu::native::OpCommand::RunOpApiV2(#atb_api, atb_call);                 \
  } while (false)

#define EXEC_ATB_CMD(atb_api, ...)                                         \
  do {                                                                     \
    const auto is_capturing =                                              \
        static_cast<int>(c10_npu::currentStreamCaptureStatusMayInitCtx()); \
    if (is_capturing) {                                                    \
      EXEC_ATB_CMD_V1(atb_api, __VA_ARGS__);                               \
    } else {                                                               \
      EXEC_ATB_CMD_V2(atb_api, __VA_ARGS__);                               \
    }                                                                      \
  } while (false)

atb::Tensor AtTensor2AtbTensor(const at::Tensor atTensor);
atb::Context* GetContext(aclrtStream stream);
uint64_t OperationSetup(atb::VariantPack variant_pack,
                        atb::Operation* operation,
                        atb::Context* context_ptr);
class ParamSetter {
 public:
  ParamSetter& Input(const at::Tensor& tensor,
                     const bool& format_trans = false);
  ParamSetter& Input(const c10::optional<at::Tensor>& tensor,
                     const bool& format_trans = false);
  ParamSetter& Output(at::Tensor& tensor);
  atb::VariantPack variant_pack_;
  TensorMaintainer tensor_maintainer_;
};

void RunAtbCmd(atb::Operation* op,
               const ParamSetter& paramsetter,
               const std::string& name);

}  // namespace atb

#endif
