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

#include "framework/kv_cache/kv_cache_impl.h"

#include "framework/kv_cache/kv_cache_utils.h"
#include "util/tensor_helper.h"

namespace xllm {

KVCacheImpl::KVCacheImpl(const KVCacheTensors& tensors)
    : key_cache_(tensors.key_cache), value_cache_(tensors.value_cache) {}

KVCacheImpl::KVCacheImpl(
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : KVCacheImpl(create_kv_cache_tensors(kv_cache_shape, create_options)) {}

torch::Tensor KVCacheImpl::get_k_cache() const { return key_cache_; }

torch::Tensor KVCacheImpl::get_v_cache() const { return value_cache_; }

std::optional<torch::Tensor> KVCacheImpl::get_k_cache_scale() const {
  return std::nullopt;
}

std::optional<torch::Tensor> KVCacheImpl::get_v_cache_scale() const {
  return std::nullopt;
}

torch::Tensor KVCacheImpl::get_index_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_conv_cache() const { return torch::Tensor(); }

torch::Tensor KVCacheImpl::get_ssm_cache() const { return torch::Tensor(); }

bool KVCacheImpl::empty() const {
  return !key_cache_.defined() || !value_cache_.defined();
}

std::vector<std::vector<int64_t>> KVCacheImpl::get_shapes() const {
  std::vector<std::vector<int64_t>> tensor_shapes(2);
  tensor_shapes[0] = get_tensor_shape(key_cache_);
  tensor_shapes[1] = get_tensor_shape(value_cache_);
  return tensor_shapes;
}

void KVCacheImpl::swap_blocks(torch::Tensor& src_tensor,
                              torch::Tensor& dst_tensor) {
  // batch select keys and values
  auto selected_keys = torch::index_select(key_cache_, 0, src_tensor);
  auto selected_values = torch::index_select(value_cache_, 0, src_tensor);

  // batch copy keys and values to dst indices
  key_cache_.index_copy_(0, dst_tensor, selected_keys);
  value_cache_.index_copy_(0, dst_tensor, selected_values);
}

}  // namespace xllm