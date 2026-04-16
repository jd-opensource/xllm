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

#include "framework/kv_cache/linear_attention_kv_cache_impl.h"

#include "util/tensor_helper.h"

namespace xllm {

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const LinearAttentionKVCacheTensors& tensors)
    : conv_cache_(tensors.conv_cache), ssm_cache_(tensors.ssm_cache) {}

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const std::vector<std::vector<int64_t>>& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : LinearAttentionKVCacheImpl(
          create_linear_attention_kv_cache_tensors(kv_cache_shape,
                                                   create_options)) {}

torch::Tensor LinearAttentionKVCacheImpl::get_conv_cache() const {
  return conv_cache_;
}

torch::Tensor LinearAttentionKVCacheImpl::get_ssm_cache() const {
  return ssm_cache_;
}

bool LinearAttentionKVCacheImpl::empty() const {
  return !conv_cache_.defined() || !ssm_cache_.defined();
}

std::vector<std::vector<int64_t>> LinearAttentionKVCacheImpl::get_shapes()
    const {
  std::vector<std::vector<int64_t>> tensor_shapes(2);
  tensor_shapes[0] = get_tensor_shape(conv_cache_);
  tensor_shapes[1] = get_tensor_shape(ssm_cache_);
  return tensor_shapes;
}

}  // namespace xllm
