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

#include "device.h"
#if defined(USE_NPU)
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#elif defined(USE_MLU)
#include <cn_api.h>
#include <torch_mlu/csrc/framework/core/device.h>
#include <torch_mlu/csrc/framework/core/device_utils.h>
#include <torch_mlu/csrc/framework/generator/generator_impl.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#endif

namespace {
#if defined(USE_CUDA)

std::pair<int32_t, int32_t> get_compute_capability(int32_t device_id) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    LOG(FATAL) << "Failed to get compute capability for device " << device_id;
  }
  return std::make_pair(prop.major, prop.minor);
}

int32_t get_cuda_version() {
  int32_t version;
  cudaError_t err = cudaRuntimeGetVersion(&version);
  if (err != cudaSuccess) {
    LOG(FATAL) << "Failed to get CUDA version!";
  }
  return version;
}

bool cuda_version_at_least(int32_t major, int32_t minor) {
  int32_t version = get_cuda_version();
  return version >= major * 1000 + minor * 10;
}

// Whether to enable Programmatic Dependent Launch (PDL). See
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
// Only supported for >= sm90, and currently only for FA2, CUDA core, and
// trtllm-gen decode.
bool support_pdl(int32_t device_id) {
  auto [major, minor] = get_compute_capability(device_id);
  return major >= 9;
}

bool support_sm90a(int32_t device_id) {
  auto [major, minor] = get_compute_capability(device_id);
  return (major == 9) && cuda_version_at_least(12, 3);
}
#endif
}  // namespace

namespace xllm {

bool Device::enable_pdl_ = false;
bool Device::support_sm90a_ = false;

Device::Device(const torch::Device& device) : device_(device) {
#if defined(USE_CUDA)
  static bool enable_pdl = support_pdl(device.index());
  enable_pdl_ = enable_pdl;

  static bool is_sm90a_supported = support_sm90a(device.index());
  support_sm90a_ = is_sm90a_supported;
#endif
}

Device::operator torch::Device() const { return unwrap(); }

void Device::set_device() const {
#if defined(USE_NPU)
  c10_npu::set_device(index());
#elif defined(USE_MLU)
  torch_mlu::setDevice(index());
#elif defined(USE_CUDA) || defined(USE_ILU)
  c10::cuda::set_device(index());
#endif
}

void Device::set_seed(uint64_t seed) const {
  torch::manual_seed(seed);
#if defined(USE_NPU)
  auto gen = at_npu::detail::getDefaultNPUGenerator(index());
  gen.set_current_seed(seed);
#elif defined(USE_MLU)
  auto gen = torch_mlu::getDefaultMLUGenerator(index());
  {
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_current_seed(seed);
  }
#elif defined(USE_CUDA)
  torch::cuda::manual_seed(seed);
#endif
}

const torch::Device& Device::unwrap() const { return device_; }

int32_t Device::index() const { return device_.index(); }

// set device before init device context
void Device::init_device_context() const {
#if defined(USE_NPU)
  torch_npu::init_npu(index());
#endif
}

int Device::device_count() {
#if defined(USE_NPU)
  return c10_npu::device_count();
#elif defined(USE_MLU)
  return torch_mlu::device_count();
#elif defined(USE_CUDA) || defined(USE_ILU)
  return c10::cuda::device_count();
#endif
}

std::string Device::type_str() {
#if defined(USE_NPU)
  return "npu";
#elif defined(USE_MLU)
  return "mlu";
#elif defined(USE_CUDA) || defined(USE_ILU)
  return "cuda";
#endif
}

torch::DeviceType Device::type_torch() {
#if defined(USE_NPU) || defined(USE_MLU)
  return torch::kPrivateUse1;
#elif defined(USE_CUDA) || defined(USE_ILU)
  return torch::kCUDA;
#endif
}

bool Device::is_enable_pdl() { return enable_pdl_; }

bool Device::is_support_sm90a() { return support_sm90a_; }

// set device before get device mem
Device::DeviceMem Device::get_device_mem() const {
  DeviceMem device_mem;
  size_t total_memory = 0;
  size_t free_memory = 0;
#if defined(USE_NPU)
  aclrtGetMemInfo(ACL_HBM_MEM, &free_memory, &total_memory);
#elif defined(USE_MLU)
  cnrtMemGetInfo(&free_memory, &total_memory);
#elif defined(USE_CUDA) || defined(USE_ILU)
  cudaMemGetInfo(&free_memory, &total_memory);
#endif
  device_mem.total_memory = static_cast<int64_t>(total_memory);
  device_mem.free_memory = static_cast<int64_t>(free_memory);
  return device_mem;
}

int64_t Device::total_memory() { return get_device_mem().total_memory; }

int64_t Device::free_memory() { return get_device_mem().free_memory; }

int Device::synchronize_default_stream() {
#if defined(USE_NPU)
  return aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(index()).stream());
#elif defined(USE_MLU)
  torch_mlu::getCurrentMLUStream(index()).synchronize();
#elif defined(USE_CUDA) || defined(USE_ILU)
  c10::cuda::getCurrentCUDAStream().synchronize();
#endif
  return 0;
}

std::unique_ptr<Stream> Device::get_stream_from_pool(const int32_t timeout) {
  return std::make_unique<Stream>(timeout);
}

}  // namespace xllm
