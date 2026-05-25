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

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/torch.h>

#include <optional>
#include <tuple>
#include <vector>

// #include "ATen/Tensor.h"

namespace xllm::kernel::dcu {

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias);

// Build a 2D block_table [B, max_pages] int32 from CSR-format paged KV
// metadata.
//
// paged_kv_indptr:  [B + 1]       int32  — cumulative page offsets per sequence
// paged_kv_indices: [total_pages]  int32  — flattened page (block) IDs
//
// Returns block_table of shape [B, total_pages] filled with page IDs and -1 for
// padding. total_pages (= indices.size(0)) is the worst-case row width, read
// from tensor metadata with zero GPU sync. The flash attention kernel only
// accesses up to the per-sequence page count, so extra -1 columns are never
// touched.
//
// Implementation:
//   - Pre-allocates [B, total_pages] and initializes to -1 via cudaMemsetAsync
//     (hardware fill engine, no kernel launch)
//   - Single kernel (one block per sequence) copies valid page IDs
torch::Tensor build_block_table_from_paged_kv_cuda(
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices);
torch::Tensor random_sample(const torch::Tensor& probs);

}  // namespace xllm::kernel::dcu
