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

#include "kv_cache.h"

namespace xllm {

KVCache::KVCache(torch::Tensor key_cache, torch::Tensor value_cache)
    : key_cache_(std::move(key_cache)), value_cache_(std::move(value_cache)) {}

torch::Tensor KVCache::get_k_cache() const { return key_cache_; }
torch::Tensor KVCache::get_v_cache() const { return value_cache_; }

void KVCache::swap_blocks(const std::vector<CacheBlockInfo>& swap_blocks) {
  if (swap_blocks.empty()) {
    return;
  }

  // collect src and dst indices
  std::vector<int64_t> src_indices, dst_indices;
  src_indices.reserve(swap_blocks.size());
  dst_indices.reserve(swap_blocks.size());

  for (const auto& block : swap_blocks) {
    src_indices.push_back(block.device_block_id);
    dst_indices.push_back(block.host_block_id);
  }

  // batch select keys and values
  auto src_tensor = torch::tensor(
      src_indices, torch::dtype(torch::kLong).device(key_cache_.device()));
  auto dst_tensor = torch::tensor(
      dst_indices, torch::dtype(torch::kLong).device(key_cache_.device()));

  // batch select keys and values
  auto selected_keys = torch::index_select(key_cache_, 0, src_tensor);
  auto selected_values = torch::index_select(value_cache_, 0, src_tensor);

  // batch copy keys and values to dst indices
  key_cache_.index_copy_(0, dst_tensor, selected_keys);
  value_cache_.index_copy_(0, dst_tensor, selected_values);
}

}  // namespace xllm
