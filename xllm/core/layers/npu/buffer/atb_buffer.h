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

#include "atb/atb_infer.h"

namespace xllm {

class AtbBuffer {
 public:
  explicit AtbBuffer(uint64_t bufferSize, at::Device device);
  ~AtbBuffer();
  void* get_buffer(uint64_t bufferSize);

 private:
  torch::Tensor create_attensor(uint64_t bufferSize) const;
  at::Tensor create_attensor_from_tensor_desc(
      const atb::TensorDesc& tensorDesc) const;

 private:
  void* buffer_ = nullptr;
  uint64_t buffer_size_ = 0;
  torch::Tensor at_tensor_;
  at::Device device_;
};

}  // namespace xllm
