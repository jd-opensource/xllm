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

#include "mm_prefix_cache.h"

#include <MurmurHash3.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <thread>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "request/mm_data_visitor.h"
#include "request/sequence.h"

namespace xllm {

void mm_murmur_hash3(const std::vector<const uint8_t*>& mm_hash_values,
                     const uint8_t* pre_hash_value,
                     const Slice<int32_t>& token_ids,
                     uint8_t* hash_value) {
  murmur_hash3(pre_hash_value, token_ids, hash_value);
  if (mm_hash_values.size() == 0) {
    return;
  }
  int32_t data_len = MURMUR_HASH3_VALUE_LEN * (mm_hash_values.size() + 1);
  std::vector<uint8_t> key_buffer;
  key_buffer.reserve(data_len);
  key_buffer.insert(
      key_buffer.end(), hash_value, hash_value + MURMUR_HASH3_VALUE_LEN);
  for (const auto* mm_hash_value : mm_hash_values) {
    key_buffer.insert(key_buffer.end(),
                      mm_hash_value,
                      mm_hash_value + MURMUR_HASH3_VALUE_LEN);
  }

  MurmurHash3_x64_128(reinterpret_cast<const void*>(key_buffer.data()),
                      key_buffer.size(),
                      FLAGS_murmur_hash3_seed,
                      hash_value);
}
MMPrefixCache::MMPrefixCache(uint32_t block_size) : PrefixCache(block_size) {}

MMPrefixCache::~MMPrefixCache() {}

std::vector<Block> MMPrefixCache::match(
    Sequence* sequence,
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks) {
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

  MMData& mm_data = sequence->mutable_mm_data();

  int32_t cur_mm_idx = 0;
  int32_t mm_num = mm_data.size();
  auto& mm_items = mm_data.items<MMItemVec>();

  for (auto& mm_item : mm_items) {
    const auto& pos = mm_item.state().token_pos();
    if (start_index >= pos.offset + pos.length)
      cur_mm_idx++;
    else
      break;
  }

  size_t end_token_index;
  size_t cur_token_index;

  for (cur_token_index = start_index; cur_token_index < n_tokens;
       cur_token_index += block_size_) {
    end_token_index = cur_token_index + block_size_;
    std::vector<const uint8_t*> mm_hash_values = get_block_mm_hash_values(
        mm_data, cur_token_index, end_token_index, cur_mm_idx);
    if (cur_token_index == 0) {
      mm_murmur_hash3(mm_hash_values,
                      nullptr,
                      token_ids.slice(cur_token_index, end_token_index),
                      token_hash_key.data);
    } else {
      mm_murmur_hash3(mm_hash_values,
                      token_hash_key.data,
                      token_ids.slice(cur_token_index, end_token_index),
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
  uint32_t last_matched_token_index = 0;
  if (cur_token_index > n_tokens) {
    cur_token_index = cur_token_index - block_size_ + 1;
  }
  last_matched_token_index =
      cur_token_index ? cur_token_index - 1 : last_matched_token_index;

  UpdateMMItemCachedStateVisitor visitor(
      last_matched_token_index, token_ids.size(), block_size_);
  mm_data.foreach (visitor);

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

size_t MMPrefixCache::insert(Sequence* sequence,
                             const Slice<int32_t>& token_ids,
                             std::vector<Block>& blocks,
                             size_t existed_shared_blocks_num,
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
  CHECK_GE(n_blocks, existed_shared_blocks_num);
  Murmur3Key token_hash_key =
      existed_shared_blocks_num == 0
          ? Murmur3Key{}
          : Murmur3Key{blocks[existed_shared_blocks_num - 1]
                           .get_immutable_hash_value()};
  int32_t cur_mm_idx = 0;
  uint32_t block_idx = existed_shared_blocks_num;
  insert_keys->reserve(n_blocks);
  const auto& mm_data = sequence->mm_data();
  const auto& mm_items = mm_data.items<MMItemVec>();
  for (auto& mm_item : mm_items) {
    const auto& pos = mm_item.state().token_pos();
    if (existed_shared_blocks_num * block_size_ >= pos.offset + pos.length) {
      cur_mm_idx++;
    } else {
      break;
    }
  }
  for (size_t i = existed_shared_blocks_num * block_size_; i < n_tokens;
       i += block_size_) {
    std::vector<const uint8_t*> mm_hash_values =
        get_block_mm_hash_values(mm_data, i, i + block_size_, cur_mm_idx);
    if (i == 0) {
      mm_murmur_hash3(mm_hash_values,
                      nullptr,
                      token_ids.slice(i, i + block_size_),
                      token_hash_key.data);
    } else {
      mm_murmur_hash3(mm_hash_values,
                      token_hash_key.data,
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
std::vector<const uint8_t*> MMPrefixCache::get_block_mm_hash_values(
    const MMData& mm_data,
    int32_t start_token_idx,
    int32_t end_token_idx,
    int32_t& start_mm_idx) {
  auto& mm_items = mm_data.items<MMItemVec>();
  std::vector<const uint8_t*> mm_hash_values;
  int32_t& cur_mm_idx = start_mm_idx;
  int32_t mm_num = mm_data.size();
  while (cur_mm_idx < mm_num) {
    const auto& mm_item_state = mm_items[cur_mm_idx].state();
    const auto& mm_pos = mm_item_state.token_pos();
    int32_t mm_start_idx = mm_pos.offset;
    int32_t mm_end_idx = mm_start_idx + mm_pos.length;
    if (end_token_idx < mm_start_idx)
      break;
    else {
      if (start_token_idx > mm_end_idx) {
        cur_mm_idx++;
        continue;
      }
      auto& prefix_cache = mm_items[cur_mm_idx].state().prefix_cache();
      mm_hash_values.push_back(prefix_cache.key.data);
      if (mm_end_idx <= end_token_idx) {
        cur_mm_idx++;
      } else {
        break;
      }
    }
  }
  return mm_hash_values;
}

}  // namespace xllm
