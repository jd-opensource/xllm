#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "common/types.h"
#include "framework/block/block.h"
#include "framework/kv_cache/kv_cache_event.h"
#include "util/slice.h"

namespace xllm {

template <typename VectorA, typename VectorB>
size_t common_prefix_length(const VectorA& token_ids1,
                            const VectorB& token_ids2) {
  size_t i = 0;
  while (i < token_ids1.size() && i < token_ids2.size() &&
         token_ids1[i] == token_ids2[i]) {
    ++i;
  }
  return i;
}

inline size_t round_down(size_t n, size_t multiple) {
  return (n / multiple) * multiple;
}

class PrefixCache {
 public:
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache(PrefixCache&&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;
  PrefixCache& operator=(PrefixCache&&) = delete;

  PrefixCache() = default;
  virtual ~PrefixCache() = default;

  std::vector<Block> match(const std::vector<int32_t>& token_ids) {
    return match(Slice<int32_t>(token_ids), {});
  }
  virtual std::vector<Block> match(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {}) = 0;

  size_t insert(const std::vector<int32_t>& token_ids,
                const std::vector<Block>& blocks) {
    return insert(Slice<int32_t>(token_ids), Slice<Block>(blocks));
  }
  virtual size_t insert(const Slice<int32_t>& token_ids,
                        const Slice<Block>& blocks) = 0;

  virtual size_t evict(size_t n_blocks) = 0;

  virtual size_t num_blocks() const = 0;

  float block_match_rate() {
    if (total_blocks_.load() == 0) {
      return 0;
    } else {
      return static_cast<float>(matched_blocks_.load()) / total_blocks_.load();
    }
  }

  virtual KvCacheEvent* get_upload_kvcache_events() {
    LOG(ERROR) << "Not implemented!";
    return nullptr;
  }

  virtual bool compute_blocks_hash_value(const Slice<int32_t>& token_ids,
                                         std::vector<Block>& blocks) {
    return false;
  }

 protected:
  std::atomic<uint64_t> total_blocks_{0}, matched_blocks_{0};
};

std::unique_ptr<PrefixCache> CreatePrefixCachePolicy(
    const int32_t block_size,
    const std::string& policy,
    const bool& enbale_service_routing = false);

}  // namespace xllm
