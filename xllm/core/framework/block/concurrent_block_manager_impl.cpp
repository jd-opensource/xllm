#include "concurrent_block_manager_impl.h"

namespace xllm {

ConcurrentBlockManagerImpl::ConcurrentBlockManagerImpl(const Options& options)
    : BlockManagerImpl(options) {}

std::vector<Block> ConcurrentBlockManagerImpl::allocate(size_t num_blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  return BlockManagerImpl::allocate(num_blocks);
}

void ConcurrentBlockManagerImpl::deallocate(const Slice<Block>& blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  BlockManagerImpl::deallocate(blocks);
}

std::vector<Block> ConcurrentBlockManagerImpl::allocate_shared(
    const Slice<int32_t>& tokens_ids,
    const Slice<Block>& existed_shared_blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  return BlockManagerImpl::allocate_shared(tokens_ids);
}

void ConcurrentBlockManagerImpl::cache(const Slice<int32_t>& token_ids,
                                       std::vector<Block>& blocks) {
  std::lock_guard<std::mutex> lock(mutex_);
  BlockManagerImpl::cache(token_ids, blocks);
}

size_t ConcurrentBlockManagerImpl::num_blocks_in_prefix_cache() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return BlockManagerImpl::num_blocks_in_prefix_cache();
}

size_t ConcurrentBlockManagerImpl::num_free_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return BlockManagerImpl::num_free_blocks();
}

double ConcurrentBlockManagerImpl::kv_cache_utilization() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return BlockManagerImpl::kv_cache_utilization();
}

}  // namespace xllm
