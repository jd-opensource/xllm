#pragma once

#include "block_manager_impl.h"

namespace xllm {

class ConcurrentBlockManagerImpl : public BlockManagerImpl {
 public:
  explicit ConcurrentBlockManagerImpl(const Options& options);
  virtual ~ConcurrentBlockManagerImpl() = default;

  // Try to allocate blocks with num_blocks,
  // return {} if not enough blocks
  std::vector<Block> allocate(size_t num_blocks) override;

  void deallocate(const Slice<Block>& blocks) override;

  // try to share blocks among sequences with the same prefix
  std::vector<Block> allocate_shared(
      const Slice<int32_t>& tokens_ids,
      const Slice<Block>& existed_shared_blocks = {}) override;

  // cache the blocks
  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks) override;

  // get the number of blocks in the prefix cache
  size_t num_blocks_in_prefix_cache() const override;

  // get the number of free blocks in the block allocator
  size_t num_free_blocks() const override;

  // get the block utilization.
  double kv_cache_utilization() const override;

 private:
  // mutex for disagg prefill/decode mode
  mutable std::mutex mutex_;
};

}  // namespace xllm
