#pragma once
#include <unordered_set>

#include "util/hash_util.h"

namespace xllm {

struct KvCacheEvent {
  std::unordered_set<Murmur3Key, FixedStringKeyHash, FixedStringKeyEqual>
      stored_cache;
  std::unordered_set<Murmur3Key, FixedStringKeyHash, FixedStringKeyEqual>
      removed_cache;
  std::unordered_set<Murmur3Key, FixedStringKeyHash, FixedStringKeyEqual>
      offload_cache;

  void clear() {
    stored_cache.clear();
    removed_cache.clear();
    offload_cache.clear();
  }
};

}  // namespace xllm
