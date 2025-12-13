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
#include <unordered_set>

#include "util/hash_util.h"

namespace xllm {

struct KvCacheEvent {
  std::unordered_set<Murmur3Key, FixedStringKeyHash, FixedStringKeyEqual>
      stored_cache;
  std::unordered_set<Murmur3Key, FixedStringKeyHash, FixedStringKeyEqual>
      removed_cache;

  void clear() {
    stored_cache.clear();
    removed_cache.clear();
  }
};

}  // namespace xllm
