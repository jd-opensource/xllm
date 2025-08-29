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

#include "prefix_cache_hash_sha256.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <string.h>

#include <iostream>
#include <thread>

#include "common/metrics.h"
#include "util/hash_util.h"

namespace xllm {

PrefixCacheHashSha256::PrefixCacheHashSha256(uint32_t block_size)
    : PrefixCacheHash(block_size), hash_value_len_(SHA256_HASH_VALUE_LEN) {}

PrefixCacheHashSha256::~PrefixCacheHashSha256() {
  LOG(INFO) << "block matched rate: " << block_match_rate();
}

std::vector<Block> PrefixCacheHashSha256::match(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks) {
  // allign tokens to block boundary
  const size_t n_tokens = round_down(token_ids.size(), block_size_);
  if (n_tokens == 0) {
    return std::vector<Block>();
  }

  const int64_t now = absl::ToUnixMicros(absl::Now());
  size_t n_blocks = n_tokens / block_size_;
  total_blocks_.fetch_add(n_blocks);

  auto tokens_slice = token_ids.slice(0, n_tokens);

  std::vector<Block> blocks;
  blocks.reserve(n_blocks);

  DNodeList node_list;

  Sha256Key token_hash_key;
  for (size_t i = 0; i < n_tokens; i += block_size_) {
    if (i == 0) {
      sha256(sha256_hash_seed(),
             token_ids.slice(i, i + block_size_),
             token_hash_key.data);
    } else {
      sha256(token_hash_key.data,
             token_ids.slice(i, i + block_size_),
             token_hash_key.data);
    }

    auto iter = sha256_cached_blocks_.find(token_hash_key);
    if (iter != sha256_cached_blocks_.end()) {
      blocks.push_back(iter->second->block);
      lru_lst_.remove_node(iter->second);
      // block_nodes.push_back(iter->second);
      node_list.push_front(iter->second);
    } else {
      break;
    }
  }

  // update LRU list
  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  matched_blocks_.fetch_add(blocks.size());

  int64_t int_rate_percent = static_cast<int64_t>(
      static_cast<double>(blocks.size()) * 100.0 / n_blocks);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_rate, int_rate_percent);
  HISTOGRAM_OBSERVE(prefix_cache_block_matched_num, blocks.size());

  return blocks;
}

size_t PrefixCacheHashSha256::insert(const Slice<int32_t>& token_ids,
                                     const Slice<Block>& blocks) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  // allign tokens to block boundary
  const size_t n_blocks =
      std::min(token_ids.size() / block_size_, blocks.size());
  const size_t n_tokens = n_blocks * block_size_;

  if (n_blocks == 0) {
    return 0;
  }

  // truncate the token ids and blocks to boundary
  auto tokens_slice = token_ids.slice(0, n_tokens);
  auto blocks_slice = blocks.slice(0, n_blocks);

  DNodeList node_list;
  Sha256Key token_hash_key;
  size_t block_idx = 0;
  for (size_t i = 0; i < n_tokens; i += block_size_) {
    if (i == 0) {
      sha256(sha256_hash_seed(),
             token_ids.slice(i, i + block_size_),
             token_hash_key.data);
    } else {
      sha256(token_hash_key.data,
             token_ids.slice(i, i + block_size_),
             token_hash_key.data);
    }

    auto iter = sha256_cached_blocks_.find(token_hash_key);
    if (iter != sha256_cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      Node* new_node = new Node();

      new_node->block = blocks[block_idx];
      new_node->block.set_hash_value(token_hash_key.data, SHA256_DIGEST_LENGTH);
      new_node->block.set_token_ids(token_ids.slice(i, i + block_size_));
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      sha256_cached_blocks_.emplace(std::make_pair(token_hash_key, new_node));

      num_blocks_++;
    }

    block_idx++;
  }

  // update LRU list
  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  return n_tokens;
}

size_t PrefixCacheHashSha256::evict(size_t n_blocks) {
  if (num_blocks_ == 0 || lru_lst_.is_empty()) {
    return 0;
  }

  size_t evict_count = 0;
  Node* iter_node = lru_lst_.get_first();
  for (; evict_count < n_blocks;) {
    if (lru_lst_.is_last(iter_node)) {
      break;
    }

    if (iter_node->block.is_shared()) {  // in use
      iter_node = iter_node->next;

      continue;
    }

    Node* del_node = iter_node;

    iter_node = lru_lst_.remove_node(del_node);

    Sha256Key token_hash_key(del_node->block.get_immutable_hash_value());

    sha256_cached_blocks_.erase(token_hash_key);

    delete del_node;

    --num_blocks_;
    ++evict_count;
  }

  return evict_count;
}

}  // namespace xllm
