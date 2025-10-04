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

#include "dit_cache.h"

namespace xllm {

bool DiTCache::init(const DiTCacheConfig& cfg) {
  active_cache_ = create_dit_cache(cfg);
  if (!active_cache_) {
    return false;
  }
  active_cache_->init(cfg);
  return true;
}

bool DiTCache::on_before_block(const CacheBlockIn& blockin) {
  return active_cache_->on_before_block(blockin);
}

CacheBlockOut DiTCache::on_after_block(const CacheBlockIn& blockin) {
  return active_cache_->on_after_block(blockin);
}

bool DiTCache::on_before_step(const CacheStepIn& stepin) {
  return active_cache_->on_before_step(stepin);
}

CacheStepOut DiTCache::on_after_step(const CacheStepIn& stepin) {
  return active_cache_->on_after_step(stepin);
}

}  // namespace xllm
