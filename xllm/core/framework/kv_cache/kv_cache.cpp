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

KVCache::KVCache(torch::Tensor key_cache,
                 torch::Tensor value_cache,
                 torch::Tensor index_cache)
    : key_cache_(std::move(key_cache)),
      value_cache_(std::move(value_cache)),
      index_cache_(std::move(index_cache)) {}

KVCache::KVCache(std::shared_ptr<XTensor> key_xtensor,
                 std::shared_ptr<XTensor> value_xtensor)
    : key_xtensor_(key_xtensor), value_xtensor_(value_xtensor) {}

torch::Tensor KVCache::get_k_cache() const { return key_cache_; }
torch::Tensor KVCache::get_v_cache() const { return value_cache_; }
torch::Tensor KVCache::get_index_cache() const { return index_cache_; }

std::vector<std::vector<int64_t>> KVCache::get_shapes() {
  std::vector<std::vector<int64_t>> tensor_shapes(3);
  if (key_cache_.defined() && key_cache_.numel() != 0) {
    std::vector<int64_t> shape;
    auto sizes = key_cache_.sizes();
    shape.resize(sizes.size());
    for (int i = 0; i < sizes.size(); ++i) {
      shape[i] = sizes[i];
    }
    tensor_shapes[0] = std::move(shape);
  }

  if (value_cache_.defined() && value_cache_.numel() != 0) {
    std::vector<int64_t> shape;
    auto sizes = value_cache_.sizes();
    shape.resize(sizes.size());
    for (int i = 0; i < sizes.size(); ++i) {
      shape[i] = sizes[i];
    }
    tensor_shapes[1] = std::move(shape);
  }

  if (index_cache_.defined() && index_cache_.numel() != 0) {
    std::vector<int64_t> shape;
    auto sizes = index_cache_.sizes();
    shape.resize(sizes.size());
    for (int i = 0; i < sizes.size(); ++i) {
      shape[i] = sizes[i];
    }
    tensor_shapes[2] = std::move(shape);
  }

  return tensor_shapes;
}

void KVCache::swap_blocks(torch::Tensor& src_tensor,
                          torch::Tensor& dst_tensor) {
  // batch select keys and values
  auto selected_keys = torch::index_select(key_cache_, 0, src_tensor);
  auto selected_values = torch::index_select(value_cache_, 0, src_tensor);

  // batch copy keys and values to dst indices
  key_cache_.index_copy_(0, dst_tensor, selected_keys);
  value_cache_.index_copy_(0, dst_tensor, selected_values);
}

std::shared_ptr<XTensor> KVCache::get_k_xtensor() const { return key_xtensor_; }
std::shared_ptr<XTensor> KVCache::get_v_xtensor() const {
  return value_xtensor_;
}
}  // namespace xllm
