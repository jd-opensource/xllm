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

#include "framework/prefix_cache/in_batch_prefix_cache.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "framework/prefix_cache/prefix_cache.h"

namespace xllm {

namespace {

bool is_same_hash(const uint8_t* lhs, const XXH3Key& rhs) {
  return std::memcmp(lhs, rhs.data, XXH3_128BITS_HASH_VALUE_LEN) == 0;
}

size_t compute_step_local_block_hashes(Sequence* sequence,
                                       size_t block_size,
                                       size_t start_block_index,
                                       std::vector<XXH3Key>* block_hashes) {
  if (sequence == nullptr || block_hashes == nullptr || block_size == 0) {
    return 0;
  }

  const size_t full_blocks = sequence->num_tokens() / block_size;
  if (full_blocks <= start_block_index) {
    block_hashes->clear();
    return 0;
  }

  block_hashes->clear();
  block_hashes->reserve(full_blocks - start_block_index);

  XXH3Key rolling_hash;
  const auto kv_blocks = sequence->kv_state().kv_blocks();
  if (start_block_index > 0) {
    if (kv_blocks.size() < start_block_index) {
      return 0;
    }
    rolling_hash.set(
        kv_blocks[start_block_index - 1].get_immutable_hash_value());
  }

  const auto tokens = sequence->tokens();
  for (size_t block_index = start_block_index; block_index < full_blocks;
       ++block_index) {
    XXH3Key current_hash;
    const auto token_block =
        tokens.slice(block_index * block_size, (block_index + 1) * block_size);
    if (block_index == 0) {
      xxh3_128bits_hash(nullptr, token_block, current_hash.data);
    } else {
      xxh3_128bits_hash(rolling_hash.data, token_block, current_hash.data);
    }
    block_hashes->push_back(current_hash);
    rolling_hash = current_hash;
  }

  return full_blocks;
}

}  // namespace

size_t InBatchPrefixCacheContext::LookupKeyHash::operator()(
    const LookupKey& key) const {
  size_t hash_value = std::hash<size_t>()(key.block_index);
  hash_value ^= FixedStringKeyHash{}(key.hash) + 0x9e3779b97f4a7c15ULL +
                (hash_value << 6) + (hash_value >> 2);
  return hash_value;
}

bool InBatchPrefixCacheContext::LookupKeyEqual::operator()(
    const LookupKey& lhs,
    const LookupKey& rhs) const {
  return lhs.block_index == rhs.block_index &&
         FixedStringKeyEqual{}(lhs.hash, rhs.hash);
}

void InBatchPrefixCacheContext::try_match(Sequence* sequence,
                                          size_t block_size) {
  if (sequence == nullptr || providers_.empty() || block_size == 0 ||
      !sequence->is_prefill_stage() || sequence->num_tokens() < block_size) {
    return;
  }

  const size_t existing_full_blocks =
      sequence->kv_state().kv_cache_tokens_num() / block_size;
  const size_t full_blocks = sequence->num_tokens() / block_size;
  if (full_blocks <= existing_full_blocks) {
    return;
  }

  std::vector<XXH3Key> consumer_hashes;
  if (compute_step_local_block_hashes(
          sequence, block_size, existing_full_blocks, &consumer_hashes) == 0 ||
      consumer_hashes.empty()) {
    return;
  }

  LookupKey lookup_key;
  lookup_key.block_index = existing_full_blocks;
  lookup_key.hash = consumer_hashes.front();

  const auto provider_iter = provider_index_.find(lookup_key);
  if (provider_iter == provider_index_.end()) {
    return;
  }

  const int32_t preferred_dp_rank = sequence->dp_rank();
  size_t best_blocks = existing_full_blocks;
  const Provider* best_provider = nullptr;

  for (size_t provider_index : provider_iter->second) {
    CHECK_LT(provider_index, providers_.size());
    const auto& provider = providers_[provider_index];
    if (provider.sequence == nullptr || provider.sequence == sequence ||
        provider.available_full_blocks <= existing_full_blocks) {
      continue;
    }
    if (preferred_dp_rank >= 0 && preferred_dp_rank != provider.dp_rank) {
      continue;
    }

    const auto provider_blocks = provider.sequence->kv_state().kv_blocks();
    if (provider_blocks.size() < provider.available_full_blocks) {
      continue;
    }

    size_t matched_blocks = existing_full_blocks;
    while (matched_blocks < full_blocks &&
           matched_blocks < provider.available_full_blocks) {
      const size_t consumer_hash_index = matched_blocks - existing_full_blocks;
      if (!is_same_hash(
              provider_blocks[matched_blocks].get_immutable_hash_value(),
              consumer_hashes[consumer_hash_index])) {
        break;
      }
      ++matched_blocks;
    }

    if (matched_blocks > best_blocks) {
      best_blocks = matched_blocks;
      best_provider = &provider;
    }
  }

  if (best_provider == nullptr || best_blocks == 0) {
    return;
  }

  const int32_t consumer_seq_id = sequence->seq_id();
  const int32_t provider_seq_id = best_provider->sequence->seq_id();
  const int32_t provider_dp_rank = best_provider->dp_rank;
  const size_t shared_blocks_before =
      sequence->kv_state().shared_kv_blocks_num();

  if (preferred_dp_rank < 0) {
    sequence->set_dp_rank(best_provider->dp_rank);
  }

  const auto provider_blocks = best_provider->sequence->kv_state().kv_blocks();
  if (provider_blocks.size() < best_blocks) {
    return;
  }

  std::vector<Block> shared_blocks;
  shared_blocks.reserve(best_blocks);
  for (size_t i = 0; i < best_blocks; ++i) {
    shared_blocks.emplace_back(provider_blocks[i]);
  }
  sequence->add_shared_kv_blocks(std::move(shared_blocks));

  LOG(INFO) << "[in_batch_prefix_cache][matched]"
            << " consumer_seq_id=" << consumer_seq_id
            << " provider_seq_id=" << provider_seq_id
            << " matched_blocks=" << best_blocks
            << " shared_blocks_before=" << shared_blocks_before
            << " shared_blocks_after="
            << sequence->kv_state().shared_kv_blocks_num()
            << " consumer_dp_rank_before=" << preferred_dp_rank
            << " consumer_dp_rank_after=" << sequence->dp_rank()
            << " provider_dp_rank=" << provider_dp_rank
            << " block_size=" << block_size;
}

void InBatchPrefixCacheContext::register_provider(
    Sequence* sequence,
    size_t block_size,
    size_t max_handle_num_tokens) {
  if (sequence == nullptr || block_size == 0 || !sequence->is_prefill_stage()) {
    return;
  }

  size_t available_full_blocks = max_handle_num_tokens / block_size;
  if (available_full_blocks == 0) {
    return;
  }
  available_full_blocks =
      std::min(available_full_blocks, sequence->kv_state().num_kv_blocks());

  auto* mutable_blocks = sequence->kv_state().mutable_kv_blocks();
  if (mutable_blocks == nullptr || mutable_blocks->empty()) {
    return;
  }
  const size_t existed_shared_blocks_num = std::min(
      sequence->kv_state().shared_kv_blocks_num(), available_full_blocks);
  const uint32_t hashed_full_blocks = PrefixCache::compute_hash_keys(
      sequence->tokens(), *mutable_blocks, existed_shared_blocks_num);
  available_full_blocks =
      std::min<size_t>(available_full_blocks, hashed_full_blocks);
  if (available_full_blocks == 0 || sequence->dp_rank() < 0) {
    return;
  }

  const size_t provider_index = providers_.size();
  providers_.push_back({sequence, sequence->dp_rank(), available_full_blocks});

  const auto blocks = sequence->kv_state().kv_blocks();
  for (size_t block_index = 0; block_index < available_full_blocks;
       ++block_index) {
    LookupKey lookup_key;
    lookup_key.block_index = block_index;
    lookup_key.hash.set(blocks[block_index].get_immutable_hash_value());
    provider_index_[lookup_key].push_back(provider_index);
  }

  LOG(INFO) << "[in_batch_prefix_cache][register_provider]"
            << " seq_id=" << sequence->seq_id()
            << " dp_rank=" << sequence->dp_rank()
            << " max_handle_num_tokens=" << max_handle_num_tokens
            << " hashed_full_blocks=" << hashed_full_blocks
            << " available_full_blocks=" << available_full_blocks
            << " existed_shared_blocks_num=" << existed_shared_blocks_num
            << " block_size=" << block_size;
}

}  // namespace xllm
