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

#include <vector>

#include "common/macros.h"

namespace xllm {
namespace page {
struct Options {
  // devices for page manager pool
  PROPERTY(std::vector<torch::Device>, devices);

  // num of layers
  PROPERTY(int64_t, num_layers) = 0;

  // total pages for page manager
  PROPERTY(int64_t, num_total_pages) = 0;

  // key or value cache size in bytes per token
  PROPERTY(int64_t, cache_size_per_token) = 0;
};
}  // namespace page
}  // namespace xllm