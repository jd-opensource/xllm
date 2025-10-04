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

#include "dit_cache_impl.h"

#include "dit_non_cache.h"
#include "fbcache.h"
#include "fbcache_taylorseer.h"
#include "taylorseer.h"

namespace xllm {

torch::Tensor DitCacheImpl::get_tensor_or_empty(const TensorMap& m,
                                                const std::string& k) {
  auto it = m.find(k);
  if (it != m.end()) return it->second;
  return torch::Tensor();
}

bool DitCacheImpl::is_similar(const torch::Tensor& lhs,
                              const torch::Tensor& rhs,
                              float threshold) {
  if (!lhs.defined() || !rhs.defined()) return false;
  if (lhs.sizes() != rhs.sizes()) return false;

  if (threshold <= 0.0f) {
    return torch::allclose(lhs, rhs);
  }

  auto diff = (lhs - rhs).abs();
  auto mean_diff = diff.mean();
  auto mean_lhs = lhs.abs().mean();

  auto rel = mean_diff / (mean_lhs + 1e-6);
  return (rel < threshold).item<bool>();
}

std::unique_ptr<DitCacheImpl> create_dit_cache(const DiTCacheConfig& cfg) {
  switch (cfg.selected_policy) {
    case PolicyType::FBCache:
      return std::make_unique<FBCache>();
    case PolicyType::TaylorSeer:
      return std::make_unique<TaylorSeer>();
    case PolicyType::FBCacheTaylorSeer:
      return std::make_unique<FBCacheTaylorSeer>();
    default:
      return std::make_unique<DiTNonCache>();
  }
}
}  // namespace xllm
