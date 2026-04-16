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

#include "framework/kv_cache/indexed_kv_cache_impl.h"

#include "util/tensor_helper.h"

namespace xllm {

IndexedKVCacheImpl::IndexedKVCacheImpl(const IndexedKVCacheTensors& tensors)
    : KVCacheImpl(tensors.kv_cache_tensors),
      index_cache_(tensors.index_cache) {}

IndexedKVCacheImpl::IndexedKVCacheImpl(
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : IndexedKVCacheImpl(
          create_indexed_kv_cache_tensors(kv_cache_shape, create_options)) {}

torch::Tensor IndexedKVCacheImpl::get_index_cache() const {
  return index_cache_;
}

bool IndexedKVCacheImpl::empty() const {
  return !key_cache_.defined() || !value_cache_.defined() ||
         !index_cache_.defined();
}

std::vector<std::vector<int64_t>> IndexedKVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> tensor_shapes(3);
  tensor_shapes[0] = get_tensor_shape(key_cache_);
  tensor_shapes[1] = get_tensor_shape(value_cache_);
  tensor_shapes[2] = get_tensor_shape(index_cache_);
  return tensor_shapes;
}

}  // namespace xllm
