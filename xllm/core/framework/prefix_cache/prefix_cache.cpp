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

#include "prefix_cache.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <string.h>

#include <iostream>
#include <thread>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "util/hash_util.h"

namespace xllm {

std::vector<Block> PrefixCache::match(
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
  blocks.insert(
      blocks.end(), existed_shared_blocks.begin(), existed_shared_blocks.end());

  DNodeList node_list;

  size_t start_index = existed_shared_blocks.size() * block_size_;
  Murmur3Key token_hash_key =
      existed_shared_blocks.empty()
          ? Murmur3Key{}
          : Murmur3Key{existed_shared_blocks.back().get_immutable_hash_value()};
  for (size_t i = start_index; i < n_tokens; i += block_size_) {
    if (i == 0) {
      murmur_hash3(
          nullptr, token_ids.slice(i, i + block_size_), token_hash_key.data);
    } else {
      murmur_hash3(token_hash_key.data,
                   token_ids.slice(i, i + block_size_),
                   token_hash_key.data);
    }

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      blocks.push_back(iter->second->block);
      lru_lst_.remove_node(iter->second);
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

size_t PrefixCache::insert(const Slice<int32_t>& token_ids,
                           std::vector<Block>& blocks) {
  std::vector<Murmur3Key> insert_keys;
  return insert(token_ids, blocks, &insert_keys);
}

size_t PrefixCache::evict(size_t n_blocks) {
  std::vector<Murmur3Key> evict_keys;
  return evict(n_blocks, &evict_keys);
}

size_t PrefixCache::insert(const Slice<int32_t>& token_ids,
                           std::vector<Block>& blocks,
                           std::vector<Murmur3Key>* insert_keys) {
  const int64_t now = absl::ToUnixMicros(absl::Now());
  // allign tokens to block boundary
  const size_t n_blocks =
      std::min(token_ids.size() / block_size_, blocks.size());
  const size_t n_tokens = n_blocks * block_size_;

  if (n_blocks == 0) {
    return 0;
  }

  // truncate the token ids and blocks to boundary

  DNodeList node_list;
  Murmur3Key token_hash_key;

  uint32_t block_idx = 0;
  insert_keys->reserve(n_blocks);
  for (size_t i = 0; i < n_tokens; i += block_size_) {
    if (i == 0) {
      murmur_hash3(
          nullptr, token_ids.slice(i, i + block_size_), token_hash_key.data);
    } else {
      murmur_hash3(token_hash_key.data,
                   token_ids.slice(i, i + block_size_),
                   token_hash_key.data);
    }
    blocks[block_idx].set_hash_value(token_hash_key.data);

    auto iter = cached_blocks_.find(token_hash_key);
    if (iter != cached_blocks_.end()) {
      iter->second->last_access_time = now;

      lru_lst_.remove_node(iter->second);
      node_list.push_front(iter->second);
    } else {
      Node* new_node = new Node();

      new_node->block = blocks[block_idx];
      new_node->last_access_time = now;

      node_list.push_front(new_node);

      cached_blocks_.emplace(std::make_pair(token_hash_key, new_node));

      num_blocks_++;

      insert_keys->emplace_back(token_hash_key.data);
    }

    ++block_idx;
  }

  while (!node_list.is_empty()) {
    Node* node = node_list.pop_front();
    lru_lst_.push_back(node);
  }

  return n_tokens;
}

size_t PrefixCache::evict(size_t n_blocks,
                          std::vector<Murmur3Key>* evict_keys) {
  if (num_blocks_ == 0 || lru_lst_.is_empty()) {
    return 0;
  }

  size_t evict_count = 0;
  Node* iter_node = lru_lst_.get_first();
  evict_keys->reserve(n_blocks);
  while (evict_count < n_blocks) {
    if (lru_lst_.is_last(iter_node)) {
      break;
    }

    if (iter_node->block.is_shared()) {  // in use
      iter_node = iter_node->next;

      continue;
    }

    Node* del_node = iter_node;

    iter_node = lru_lst_.remove_node(del_node);

    Murmur3Key token_hash_key(del_node->block.get_immutable_hash_value());

    cached_blocks_.erase(token_hash_key);

    delete del_node;
    ++evict_count;
    --num_blocks_;

    evict_keys->emplace_back(std::move(token_hash_key));
  }

  return evict_count;
}

uint32_t PrefixCache::compute_hash_keys(const Slice<int32_t>& token_ids,
                                        std::vector<Block>& blocks) {
  if (blocks.size() == 0) {
    return 0;
  }
  int32_t block_size = blocks[0].size();
  const size_t n_blocks = (token_ids.size() + block_size - 1) / block_size;
  if (blocks.size() > n_blocks) {
    LOG(ERROR) << "token ids do not cover the allocate block.";
    return 0;
  }

  for (size_t i = 0; i < token_ids.size() / block_size; i++) {
    if (i == 0) {
      murmur_hash3(nullptr,
                   token_ids.slice(i * block_size, (i + 1) * block_size),
                   blocks[i].get_mutable_hash_value());
    } else {
      murmur_hash3(blocks[i - 1].get_mutable_hash_value(),
                   token_ids.slice(i * block_size, (i + 1) * block_size),
                   blocks[i].get_mutable_hash_value());
    }
  }

  return token_ids.size() / block_size;
}

}  // namespace xllm
