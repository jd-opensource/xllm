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

// clang-format off
#if defined(USE_NPU)
#include "graph/types.h"
#endif
// clang-format on

#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>

#include <cstdint>
#if defined(USE_NPU)
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/torch_npu.h>
#elif defined(USE_MLU)
#include <torch_mlu/csrc/framework/core/MLUStream.h>
#elif defined(USE_CUDA)
#include <c10/cuda/CUDAStream.h>
#endif

namespace xllm {

class Stream {
 public:
  Stream();
  ~Stream() = default;

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&&) = default;
  Stream& operator=(Stream&&) = default;

  int synchronize() const;
  c10::StreamGuard set_stream_guard() const;

 private:
#if defined(USE_NPU)
  c10_npu::NPUStream stream_;
#elif defined(USE_MLU)
  torch_mlu::MLUStream stream_;
#elif defined(USE_CUDA)
  c10::cuda::CUDAStream stream_;
#endif
};

}  // namespace xllm