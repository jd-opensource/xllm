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

#include "framework/kv_cache/kv_cache_shape.h"
#include "util/tensor_helper.h"

namespace xllm {

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const LinearAttentionKVCacheTensors& tensors)
    : conv_cache_(tensors.conv_cache),
      ssm_cache_(tensors.ssm_cache),
      conv_cache_shape_(get_tensor_shape(tensors.conv_cache)),
      ssm_cache_shape_(get_tensor_shape(tensors.ssm_cache)) {}

LinearAttentionKVCacheImpl::LinearAttentionKVCacheImpl(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options)
    : LinearAttentionKVCacheImpl(
          create_linear_attention_kv_cache_tensors(kv_cache_shape,
                                                   create_options)) {
  conv_cache_shape_ = kv_cache_shape.conv_cache_shape();
  ssm_cache_shape_ = kv_cache_shape.ssm_cache_shape();
}

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
  std::vector<std::vector<int64_t>> shapes;
  shapes.reserve(2);
  shapes.emplace_back(conv_cache_shape_);
  shapes.emplace_back(ssm_cache_shape_);
  return shapes;
}

}  // namespace xllm
