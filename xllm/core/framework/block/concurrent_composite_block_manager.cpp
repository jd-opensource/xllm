/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "concurrent_composite_block_manager.h"

namespace xllm {

ConcurrentCompositeBlockManager::ConcurrentCompositeBlockManager(
    const std::vector<CacheGroupSpec>& specs)
    : composite_(specs) {}

bool ConcurrentCompositeBlockManager::allocate(BlockManagerContext* context,
                                               size_t num_tokens) {
  std::lock_guard<std::mutex> lock(mutex_);
  return composite_.allocate(context, num_tokens);
}

void ConcurrentCompositeBlockManager::deallocate(BlockManagerContext* context) {
  std::lock_guard<std::mutex> lock(mutex_);
  composite_.deallocate(context);
}

CompositeMatchResult ConcurrentCompositeBlockManager::match_prefix_cache(
    BlockManagerContext* context,
    const Slice<int32_t>& tokens,
    const MMData* mm_data) {
  std::lock_guard<std::mutex> lock(mutex_);
  return composite_.match_prefix_cache(context, tokens, mm_data);
}

size_t ConcurrentCompositeBlockManager::num_free_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return composite_.num_free_blocks();
}

size_t ConcurrentCompositeBlockManager::group_free_blocks(
    CacheStateId state_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return composite_.group_free_blocks(state_id);
}

size_t ConcurrentCompositeBlockManager::num_used_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return composite_.num_used_blocks();
}

size_t ConcurrentCompositeBlockManager::num_total_blocks() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return composite_.num_total_blocks();
}

size_t ConcurrentCompositeBlockManager::num_blocks_in_prefix_cache() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return composite_.num_blocks_in_prefix_cache();
}

double ConcurrentCompositeBlockManager::kv_cache_utilization() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return composite_.kv_cache_utilization();
}

}  // namespace xllm
