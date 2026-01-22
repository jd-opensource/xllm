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

#include "utils.h"

#include <c10/core/Device.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "core/platform/device.h"
#include "core/util/env_var.h"

namespace {
const std::unordered_map<torch::ScalarType, std::string_view>
    filename_safe_dtype_map = {
        {torch::kFloat16, "f16"},
        {torch::kBFloat16, "bf16"},
        {torch::kFloat8_e4m3fn, "e4m3"},
        {torch::kFloat8_e5m2, "e5m2"},
        {torch::kInt8, "i8"},
        {torch::kUInt8, "u8"},
        {torch::kInt32, "i32"},
        {torch::kUInt32, "u32"},
        {torch::kInt64, "i64"},
        {torch::kUInt64, "u64"},
};

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
      LOG(FATAL) << "Unsupported scalar type: ";
      break;
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
      LOG(FATAL) << "Cannot pack tensors on " << device.str();
      break;
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
}  // namespace

namespace xllm::kernel::cuda {

bool support_pdl() { return Device::is_enable_pdl(); }

std::string path_to_uri_so_lib(const std::string& uri) {
  return util::get_string_env("FLASHINFER_OPS_PATH") + "/" + uri + "/" + uri +
         ".so";
}

std::string determine_attention_backend(int64_t pos_encoding_mode,
                                        bool use_fp16_qk_reduction,
                                        bool use_custom_mask) {
  bool support_fa3_backend =
      (pos_encoding_mode == 0) && !use_fp16_qk_reduction && !use_custom_mask;

  if (Device::is_support_sm90a() && support_fa3_backend) {
    return "fa3";
  }
  return "fa2";
}

std::string get_batch_prefill_uri(const std::string& backend,
                                  torch::ScalarType dtype_q,
                                  torch::ScalarType dtype_kv,
                                  torch::ScalarType dtype_o,
                                  torch::ScalarType dtype_idx,
                                  int64_t head_dim_qk,
                                  int64_t head_dim_vo,
                                  int64_t pos_encoding_mode,
                                  bool use_sliding_window,
                                  bool use_logits_soft_cap,
                                  bool use_fp16_qk_reduction) {
  std::ostringstream oss;
  oss << "batch_prefill_with_kv_cache_"
      << "dtype_q_" << filename_safe_dtype_map.at(dtype_q) << "_"
      << "dtype_kv_" << filename_safe_dtype_map.at(dtype_kv) << "_"
      << "dtype_o_" << filename_safe_dtype_map.at(dtype_o) << "_"
      << "dtype_idx_" << filename_safe_dtype_map.at(dtype_idx) << "_"
      << "head_dim_qk_" << head_dim_qk << "_"
      << "head_dim_vo_" << head_dim_vo << "_"
      << "posenc_" << pos_encoding_mode << "_"
      << "use_swa_" << (use_sliding_window ? "True" : "False") << "_"
      << "use_logits_cap_" << (use_logits_soft_cap ? "True" : "False") << "_"
      << "f16qk_" << (use_fp16_qk_reduction ? "True" : "False");

  if (backend == "fa3") oss << "_sm90";

  return oss.str();
}

std::string get_batch_decode_uri(torch::ScalarType dtype_q,
                                 torch::ScalarType dtype_kv,
                                 torch::ScalarType dtype_o,
                                 torch::ScalarType dtype_idx,
                                 int64_t head_dim_qk,
                                 int64_t head_dim_vo,
                                 int64_t pos_encoding_mode,
                                 bool use_sliding_window,
                                 bool use_logits_soft_cap) {
  std::ostringstream oss;
  oss << "batch_decode_with_kv_cache_"
      << "dtype_q_" << filename_safe_dtype_map.at(dtype_q) << "_"
      << "dtype_kv_" << filename_safe_dtype_map.at(dtype_kv) << "_"
      << "dtype_o_" << filename_safe_dtype_map.at(dtype_o) << "_"
      << "dtype_idx_" << filename_safe_dtype_map.at(dtype_idx) << "_"
      << "head_dim_qk_" << head_dim_qk << "_"
      << "head_dim_vo_" << head_dim_vo << "_"
      << "posenc_" << pos_encoding_mode << "_"
      << "use_swa_" << (use_sliding_window ? "True" : "False") << "_"
      << "use_logits_cap_" << (use_logits_soft_cap ? "True" : "False");

  return oss.str();
}

// torch tensor is only on cpu
torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device) {
  static std::unordered_map<std::string, torch::Tensor> cache_buffer_map;
  int32_t seq_len_pow2 = xllm::util::ceil_pow2(seq_len);

  std::string key = std::string("range_") + std::to_string(seq_len_pow2);
  auto it = cache_buffer_map.find(key);
  if (it != cache_buffer_map.end()) {
    return it->second.slice(0, 0, seq_len);
  }

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor buffer = torch::arange(seq_len_pow2, options);
  cache_buffer_map.insert(std::make_pair(key, buffer));
  return buffer.slice(0, 0, seq_len);
}

std::tuple<torch::Tensor, double> split_scale_param(
    const torch::Tensor& scale) {
  if (!scale.defined()) {
    return std::make_tuple(torch::Tensor(), 1.0);
  }

  if (scale.dim() == 0) {
    return std::make_tuple(torch::Tensor(), scale.item<double>());
  }

  return std::make_tuple(scale, 1.0);
}

ffi::Tensor to_ffi_tensor(const torch::Tensor& torch_tensor) {
  if (!torch_tensor.defined()) {
    LOG(FATAL) << "torch_tensor is not defined";
  }

  auto dlpack = toDLPackImpl<DLManagedTensorVersioned>(torch_tensor);
  return ffi::Tensor::FromDLPackVersioned(dlpack);
}

ffi::Module get_module(const std::string& uri) {
  static std::unordered_map<std::string, ffi::Module> module_cache;
  static std::mutex mtx;

  std::lock_guard<std::mutex> lock(mtx);
  auto it = module_cache.find(uri);
  if (it != module_cache.end()) {
    return it->second;
  }

  std::string so_file_path = path_to_uri_so_lib(uri);
  auto mod = ffi::Module::LoadFromFile(so_file_path);
  module_cache.emplace(uri, mod);
  return mod;
}

ffi::Function get_function(const std::string& uri,
                           const std::string& func_name) {
  static std::unordered_map<std::string, ffi::Function> func_cache;
  static std::mutex mtx;

  std::string key = uri + "|" + func_name;

  std::lock_guard<std::mutex> lock(mtx);
  auto it = func_cache.find(key);
  if (it != func_cache.end()) {
    return it->second;
  }

  auto func = get_module(uri)->GetFunction(func_name).value();
  func_cache.emplace(key, func);
  return func;
}
}  // namespace xllm::kernel::cuda
