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

#ifndef XLLM_CORE_KERNELS_NPU_ATB_COMMON_H
#define XLLM_CORE_KERNELS_NPU_ATB_COMMON_H
#include <dlfcn.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>

#include "./operation_create.h"
#include "atb/atb_infer.h"
#include "utils.h"

namespace atb {

using aclTensor = struct aclTensor;
constexpr int64_t MAX_DIM_NUM = 5;
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

#define GET_OP_API_FUNC(api_name) \
  reinterpret_cast<_##api_name>(get_api_func_addr(#api_name))

inline const char* get_atb_api_lib_name(void) { return "libatb.so"; }

inline const char* get_op_api_lib_name(void) { return "libopapi.so"; }

inline void* get_api_lib_handler(const char* lib_name) {
  auto handler = dlopen(lib_name, RTLD_LAZY);
  if (handler == nullptr) {
    ASCEND_LOGW("dlopen %s failed, error:%s.", lib_name, dlerror());
  }
  return handler;
}

inline void* get_api_func_addr_in_lib(void* handler,
                                      const char* lib_name,
                                      const char* api_name) {
  auto func_addr = dlsym(handler, api_name);
  if (func_addr == nullptr) {
    ASCEND_LOGW(
        "dlsym %s from %s failed, error:%s.", api_name, lib_name, dlerror());
  }
  return func_addr;
}

inline void* get_api_func_addr(const char* api_name) {
  static auto atb_api_handler = get_api_lib_handler(get_atb_api_lib_name());
  if (atb_api_handler != nullptr) {
    auto func_addr = get_api_func_addr_in_lib(
        atb_api_handler, get_atb_api_lib_name(), api_name);
    if (func_addr != nullptr) {
      return func_addr;
    }
  }
  static auto op_api_handler = get_api_lib_handler(get_op_api_lib_name());
  if (op_api_handler != nullptr) {
    auto func_addr = get_api_func_addr_in_lib(
        op_api_handler, get_op_api_lib_name(), api_name);
    if (func_addr != nullptr) {
      return func_addr;
    }
    TORCH_CHECK(false, "get_api_func_addr not found ", api_name);
  }
}

struct TensorMaintainer {
  c10::SmallVector<at::Tensor, N>
      contiguous_tensors;  // npu tensor's life should maintain when
                           // uncontiguous to contiguous.
  c10::SmallVector<at::Tensor, N>
      cpu_tensors;  // cpu tensor's life should maintain in taskqueue.
};

inline aclTensor* convert_type(TensorMaintainer& maintainer,
                               const at::Tensor& tensor) {
  static const auto aclCreateTensor =
      reinterpret_cast<_aclCreateTensor>(get_api_func_addr("aclCreateTensor"));
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (!tensor.defined()) {
    return nullptr;
  }
  at::Tensor at_tensor = tensor.contiguous();
  aclFormat format = atb::utils::get_format_for_atb(at_tensor);

  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_data_type =
      atb::utils::convert_to_acl_data_type(scalar_data_type);
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

inline aclTensor* convert_type(TensorMaintainer& maintainer,
                               const c10::optional<at::Tensor>& opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return convert_type(maintainer, opt_tensor.value());
  }

  return nullptr;
}

template <typename T>
T convert_type(TensorMaintainer& maintainer, T value) {
  return value;
}

template <typename... Ts>
constexpr auto convert_types(TensorMaintainer& maintainer, Ts&... args) {
  return std::make_tuple(convert_type(maintainer, args)...);
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

inline TensorStructPtr copy_type_v2(TensorMaintainer& maintainer,
                                    const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return nullptr;
  }
  at::Tensor at_tensor = tensor.contiguous();
  aclFormat format = atb::utils::get_format_for_atb(at_tensor);
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

inline TensorStructPtr copy_type_v2(
    TensorMaintainer& maintainer,
    const c10::optional<at::Tensor>& opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return copy_type_v2(maintainer, opt_tensor.value());
  }

  return nullptr;
}

template <typename T>
T copy_type_v2(TensorMaintainer& maintainer, T value) {
  return value;
}

inline aclTensor* convert_type_v2(TensorStructPtr at_tensor) {
  static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
  if (aclCreateTensor == nullptr) {
    return nullptr;
  }

  if (at_tensor == nullptr) {
    return nullptr;
  }
  at::ScalarType scalar_data_type = (*at_tensor).scalar_type;
  aclDataType acl_data_type =
      atb::utils::convert_to_acl_data_type(scalar_data_type);
  c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
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
T convert_type_v2(T value) {
  return value;
}

template <typename Tuple, std::size_t... I>
auto convert_types_impl_v2(const Tuple& t, std::index_sequence<I...>) {
  return std::make_tuple(convert_type_v2(std::get<I>(t))...);
}

template <typename... Ts>
constexpr auto convert_types_v2(const std::tuple<Ts...>& args,
                                uint64_t* workspace_size_addr,
                                atb::Operation** op_addr,
                                atb::Context* context_ptr) {
  auto convert_args =
      convert_types_impl_v2(args, std::make_index_sequence<sizeof...(Ts)>{});
  auto appends = std::make_tuple(workspace_size_addr, op_addr, context_ptr);
  return std::tuple_cat(convert_args, appends);
}

template <typename... Ts>
constexpr auto copy_types_v2(TensorMaintainer& maintainer, Ts&... args) {
  return std::make_tuple(copy_type_v2(maintainer, args)...);
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
auto convert_to_op_api_func(const Tuple& params,
                            void* opApiAddr,
                            std::index_sequence<I...>) {
  using OpApiFunc =
      int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
  return func;
}

template <typename Tuple>
auto convert_to_op_api_func(const Tuple& params, void* opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return convert_to_op_api_func(
      params, opApiAddr, std::make_index_sequence<size>{});
}

inline void release(atb::Context* context) {}

inline void release(aclTensor* p) {
  static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
  if (aclDestroyTensor == nullptr) {
    return;
  }
  aclDestroyTensor(p);
}

template <typename T>
void release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void call_release(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void release_convert_types(Tuple& t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  call_release(t, std::make_index_sequence<size>{});
}

#define EXEC_ATB_CMD_V1(atb_api, ...)                                        \
  do {                                                                       \
    static const auto getWorkspaceSizeFuncAddr =                             \
        get_api_func_addr(#atb_api "GetWorkspaceSize");                      \
    static const auto atbApiFuncAddr = get_api_func_addr(#atb_api);          \
    TORCH_CHECK(                                                             \
        getWorkspaceSizeFuncAddr != nullptr && atbApiFuncAddr != nullptr,    \
        #atb_api,                                                            \
        " or ",                                                              \
        #atb_api "GetWorkspaceSize",                                         \
        " not in ",                                                          \
        get_atb_api_lib_name(),                                              \
        ", or ",                                                             \
        get_atb_api_lib_name(),                                              \
        "not found.");                                                       \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);          \
    auto context_ptr = atb::utils::get_context(acl_stream);                  \
    uint64_t workspace_size = 0;                                             \
    uint64_t* workspace_size_addr = &workspace_size;                         \
    atb::Operation* op = nullptr;                                            \
    atb::Operation** op_addr = &op;                                          \
    TensorMaintainer tensor_maintainer;                                      \
    auto converted_params = convert_types(tensor_maintainer,                 \
                                          __VA_ARGS__,                       \
                                          workspace_size_addr,               \
                                          op_addr,                           \
                                          context_ptr);                      \
    static auto getWorkspaceSizeFunc =                                       \
        convert_to_op_api_func(converted_params, getWorkspaceSizeFuncAddr);  \
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
      release_convert_types(converted_params);                               \
      return api_ret;                                                        \
    };                                                                       \
    at_npu::native::OpCommand::RunOpApiV2(#atb_api, atb_call);               \
  } while (false)

#define EXEC_ATB_CMD_V2(atb_api, ...)                                          \
  do {                                                                         \
    static const auto getWorkspaceSizeFuncAddr =                               \
        get_api_func_addr(#atb_api "GetWorkspaceSize");                        \
    static const auto AtbApiFuncAddr = get_api_func_addr(#atb_api);            \
    TORCH_CHECK(                                                               \
        getWorkspaceSizeFuncAddr != nullptr && AtbApiFuncAddr != nullptr,      \
        #atb_api,                                                              \
        " or ",                                                                \
        #atb_api "GetWorkspaceSize",                                           \
        " not in ",                                                            \
        get_atb_api_lib_name(),                                                \
        ", or ",                                                               \
        get_atb_api_lib_name(),                                                \
        "not found.");                                                         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);            \
    TensorMaintainer tensor_maintainer;                                        \
    auto copied_params = copy_types_v2(tensor_maintainer, __VA_ARGS__);        \
    auto hash_id = compute_hash(std::string(#atb_api), __VA_ARGS__);           \
    const c10::SmallVector<at::Tensor, N>& cpu_tensors =                       \
        tensor_maintainer.cpu_tensors;                                         \
    auto atb_call =                                                            \
        [copied_params, acl_stream, hash_id, cpu_tensors]() -> int {           \
      auto context_ptr = atb::utils::get_context(acl_stream);                  \
      uint64_t workspace_size = 0;                                             \
      uint64_t* workspace_size_addr = &workspace_size;                         \
      OpParamCache<uint64_t>& opParamCache =                                   \
          OpParamCache<uint64_t>::getInstance();                               \
      atb::Operation* op = opParamCache.get_operation(hash_id);                \
      atb::Operation** op_addr = &op;                                          \
      int api_ret = 0;                                                         \
      auto converted_params = convert_types_v2(                                \
          copied_params, workspace_size_addr, op_addr, context_ptr);           \
      auto getWorkspaceSizeFunc =                                              \
          convert_to_op_api_func(converted_params, getWorkspaceSizeFuncAddr);  \
      auto workspace_status = call(getWorkspaceSizeFunc, converted_params);    \
      opParamCache.save_operation(hash_id, op);                                \
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
      release_convert_types(converted_params);                                 \
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

atb::Tensor at_tensor_to_atb_tensor(const at::Tensor atTensor);
atb::Context* get_context(aclrtStream stream);
uint64_t operation_setup(atb::VariantPack variant_pack,
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

void run_atb_cmd(atb::Operation* op,
                 const ParamSetter& paramsetter,
                 const std::string& name);

}  // namespace atb

#endif
