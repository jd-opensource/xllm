#pragma once

#include <glog/logging.h>

#include "prefix_cache.h"
#include "util/double_buffer.h"

namespace xllm {
class PrefixCacheWithUpload final : public PrefixCache {
 public:
  explicit PrefixCacheWithUpload(uint32_t block_size);

  ~PrefixCacheWithUpload();

  // insert the token ids and blocks into the prefix tree
  // and set hash key to the corresponding block
  // return the length of new inserted tokens
  size_t insert(const Slice<int32_t>& token_ids,
                std::vector<Block>& blocks) override;

  // insert the blocks with hash key into the prefix tree
  size_t insert(const std::vector<Block>& blocks) override;
  size_t insert(Slice<Block>& blocks) override;

  // evict blocks hold by the prefix cache
  // return the actual number of evicted blocks
  size_t evict(size_t n_blocks) override;

  virtual KvCacheEvent* get_upload_kvcache_events() override;

 private:
  void save_event_async(const bool is_insert, std::vector<Murmur3Key>& keys);

 private:
  ThreadPool threadpool_;

  std::mutex mutex_;
  DoubleBuffer<KvCacheEvent> db_kvcache_events_;
};

}  // namespace xllm
