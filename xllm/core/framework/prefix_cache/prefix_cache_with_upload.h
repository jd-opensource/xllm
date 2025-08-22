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

  // evict blocks hold by the prefix cache
  // return the actual number of evicted blocks
  size_t evict(size_t n_blocks) override;

  virtual KvCacheEvent* get_upload_kvcache_events() override;

 private:
  ThreadPool threadpool_;

  DoubleBuffer<KvCacheEvent> db_kvcache_events_;
};

}  // namespace xllm
