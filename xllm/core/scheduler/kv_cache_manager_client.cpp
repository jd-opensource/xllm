/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "kv_cache_manager_client.h"

#include <glog/logging.h>

namespace xllm {

KVCacheManagerClient::KVCacheManagerClient(BlockManagerPool* block_manager_pool)
    : block_manager_pool_(block_manager_pool) {}

KVCacheManagerClient::KVCacheManagerClient(PageManagerPool* page_manager_pool)
    : page_manager_pool_(page_manager_pool) {}

bool KVCacheManagerClient::allocate(Sequence* sequence) {
  if (block_manager_pool_ != nullptr) {
    return block_manager_pool_->allocate(sequence);
  }
  return page_manager_pool_->allocate(sequence);
}

bool KVCacheManagerClient::allocate(std::vector<Sequence*>& sequences) {
  if (block_manager_pool_ != nullptr) {
    return block_manager_pool_->allocate(sequences);
  }
  return page_manager_pool_->allocate(sequences);
}

bool KVCacheManagerClient::allocate(Sequence* sequence, size_t num_tokens) {
  if (block_manager_pool_ != nullptr) {
    return block_manager_pool_->allocate(sequence, num_tokens);
  }
  return page_manager_pool_->allocate(sequence, num_tokens);
}

uint32_t KVCacheManagerClient::pre_allocate(Sequence* sequence) {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->pre_allocate(sequence);
}

std::vector<Block> KVCacheManagerClient::allocate(size_t num_tokens,
                                                  int32_t& dp_rank) {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->allocate(num_tokens, dp_rank);
}

void KVCacheManagerClient::deallocate(Request* request) {
  if (block_manager_pool_ != nullptr) {
    block_manager_pool_->deallocate(request);
  } else {
    page_manager_pool_->deallocate(request);
  }
}

void KVCacheManagerClient::deallocate(std::vector<Sequence*>& sequences) {
  if (block_manager_pool_ != nullptr) {
    block_manager_pool_->deallocate(sequences);
  } else {
    page_manager_pool_->deallocate(sequences);
  }
}

void KVCacheManagerClient::deallocate(Sequence* sequence) {
  if (block_manager_pool_ != nullptr) {
    block_manager_pool_->deallocate(sequence);
  } else {
    page_manager_pool_->deallocate(sequence);
  }
}

void KVCacheManagerClient::allocate_shared(Sequence* sequence) {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->allocate_shared(sequence);
}

void KVCacheManagerClient::cache(Sequence* sequence) {
  if (block_manager_pool_ != nullptr) {
    block_manager_pool_->cache(sequence);
  } else {
    page_manager_pool_->cache(sequence);
  }
}

std::vector<std::vector<CacheBlockInfo>>*
KVCacheManagerClient::get_copy_in_cache_block_infos() {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->get_copy_in_cache_block_infos();
}

std::vector<std::vector<CacheBlockInfo>>*
KVCacheManagerClient::get_copy_out_cache_block_infos() {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->get_copy_out_cache_block_infos();
}

void KVCacheManagerClient::reset_copy_content() {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  block_manager_pool_->reset_copy_content();
}

std::vector<size_t> KVCacheManagerClient::num_blocks_in_prefix_cache() const {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->num_blocks_in_prefix_cache();
}

std::vector<size_t> KVCacheManagerClient::num_free_blocks() const {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->num_free_blocks();
}

std::vector<size_t> KVCacheManagerClient::num_used_blocks() const {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->num_used_blocks();
}

double KVCacheManagerClient::kv_cache_utilization() const {
  if (block_manager_pool_ != nullptr) {
    return block_manager_pool_->kv_cache_utilization();
  }
  return page_manager_pool_->kv_cache_utilization();
}

const BlockManagerPool::Options& KVCacheManagerClient::options() const {
  if (block_manager_pool_ == nullptr) {
    LOG(FATAL) << "PageManagerPool is not supported";
  }
  return block_manager_pool_->options();
}
}  // namespace xllm