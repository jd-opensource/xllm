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

#include <torch/torch.h>

namespace xllm::kernel::cuda {

inline bool is_torch_musa_device(const torch::Device& device) {
#if defined(XLLM_TORCH_MUSA)
  // MUSA builds use USE_CUDA=ON; some tensors still report as kCUDA.
  return device.is_privateuseone() || device.is_cuda();
#else
  return device.is_privateuseone();
#endif
}

void bind_musa_tvmffi_stream(const torch::Device& device);

void sync_current_musa_stream(const torch::Device& device);

void sync_musa_ffi_stream(const torch::Device& device);

// RAII: flush torch compute, bind Mate/TVM-FFI to a pooled stream, sync after.
class MusaTvmffiStreamGuard {
 public:
  explicit MusaTvmffiStreamGuard(const torch::Device& device);
  ~MusaTvmffiStreamGuard();

  MusaTvmffiStreamGuard(const MusaTvmffiStreamGuard&) = delete;
  MusaTvmffiStreamGuard& operator=(const MusaTvmffiStreamGuard&) = delete;

 private:
  torch::Device device_;
  bool active_ = false;
};

}  // namespace xllm::kernel::cuda
