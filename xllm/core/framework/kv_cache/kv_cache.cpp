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

KVCache::KVCache(std::shared_ptr<XTensor> key_xtensor,
                 std::shared_ptr<XTensor> value_xtensor)
    : key_xtensor_(key_xtensor), value_xtensor_(value_xtensor) {}

torch::Tensor KVCache::get_k_cache() const { return key_cache_; }
torch::Tensor KVCache::get_v_cache() const { return value_cache_; }

std::shared_ptr<XTensor> KVCache::get_k_xtensor() const { return key_xtensor_; }
std::shared_ptr<XTensor> KVCache::get_v_xtensor() const {
  return value_xtensor_;
}
}  // namespace xllm
