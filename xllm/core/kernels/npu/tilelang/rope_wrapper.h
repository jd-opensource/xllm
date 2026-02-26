/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

namespace xllm::kernel::npu::tilelang {

// Apply TileLang RoPE kernel in-place on a single input tensor.
// Invalid inputs trigger CHECK failures.
// Supports input not contiguous, with stride.
void rope_in_place(torch::Tensor& input,
                   const torch::Tensor& sin_cache,
                   const torch::Tensor& cos_cache);

}  // namespace xllm::kernel::npu::tilelang
