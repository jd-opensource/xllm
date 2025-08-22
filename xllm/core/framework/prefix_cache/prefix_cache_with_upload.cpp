
#include "prefix_cache_with_upload.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <string.h>

#include <iostream>
#include <thread>

#include "common/metrics.h"
#include "util/hash_util.h"

namespace xllm {

PrefixCacheWithUpload::PrefixCacheWithUpload(uint32_t block_size)
    : PrefixCache(block_size) {
  db_kvcache_events_.set_front_value(new KvCacheEvent());
  db_kvcache_events_.set_back_value(new KvCacheEvent());
}

PrefixCacheWithUpload::~PrefixCacheWithUpload() {
  auto back = db_kvcache_events_.get_back_value();
  if (back) {
    delete back;
  }

  auto front = db_kvcache_events_.get_back_value();
  if (front) {
    delete front;
  }
}

size_t PrefixCacheWithUpload::insert(const Slice<int32_t>& token_ids,
                                     std::vector<Block>& blocks) {
  std::vector<Murmur3Key> insert_keys;
  auto n_tokens = PrefixCache::insert(token_ids, blocks, &insert_keys);

  threadpool_.schedule([insert_keys = std::move(insert_keys), this]() {
    auto front_ptr = this->db_kvcache_events_.get_front_value();
    if (!front_ptr) {
      LOG(INFO) << "Front DoubleBufferKvCacheEvent is nullptr!";
      return;
    }
    if (!this->exited_.load()) {
      for (const auto& hash_id : insert_keys) {
        front_ptr->removed_cache.erase(hash_id);
        front_ptr->stored_cache.insert(hash_id);
      }
    }
  });

  return n_tokens;
}

size_t PrefixCacheWithUpload::evict(size_t n_blocks) {
  std::vector<Murmur3Key> evict_keys;
  auto evict_count = PrefixCache::evict(n_blocks, &evict_keys);

  threadpool_.schedule([evict_keys = std::move(evict_keys), this]() {
    auto front_ptr = this->db_kvcache_events_.get_front_value();
    if (!front_ptr) {
      LOG(INFO) << "Front DoubleBufferKvCacheEvent is nullptr!";
      return;
    }
    if (!this->exited_.load()) {
      for (const auto& hash_id : evict_keys) {
        front_ptr->removed_cache.insert(hash_id);
        front_ptr->stored_cache.erase(hash_id);
      }
    }
  });

  return evict_count;
}

KvCacheEvent* PrefixCacheWithUpload::get_upload_kvcache_events() {
  db_kvcache_events_.swap();
  if (!exited_.load()) {
    return db_kvcache_events_.get_back_value();
  } else {
    return nullptr;
  }
}

}  // namespace xllm
