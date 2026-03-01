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

#include "embedding_cache.h"

#include <glog/logging.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace xllm {

EmbeddingCache::EmbeddingCache(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No embeddings to allocate";
  decode_tails_.resize(total_nums);
}

// write embeddings to cache
void EmbeddingCache::write(int32_t id, const torch::Tensor& embeddings) {
  set_last_state(id, embeddings, /*token_id=*/-1);
}

void EmbeddingCache::write(const std::vector<int32_t>& ids,
                           const torch::Tensor& embeddings) {
  int32_t total_nums = ids.size();
  CHECK_EQ(total_nums, embeddings.size(0));
  for (int32_t i = 0; i < total_nums; ++i) {
    write(ids[i], embeddings[i]);
  }
}

void EmbeddingCache::write_validate(const std::vector<int32_t>& ids,
                                    const torch::Tensor& next_tokens,
                                    const torch::Tensor& embeddings,
                                    int32_t num_speculative_tokens) {
  CHECK_EQ(next_tokens.dim(), 2) << "next_tokens should be [batch, num_tokens]";
  CHECK_EQ(embeddings.dim(), 3)
      << "embeddings should be [batch, num_tokens, h]";
  CHECK_EQ(next_tokens.size(0), static_cast<int64_t>(ids.size()))
      << "next_tokens batch mismatch";
  CHECK_EQ(embeddings.size(0), static_cast<int64_t>(ids.size()))
      << "embeddings batch mismatch";
  CHECK_EQ(next_tokens.size(1), embeddings.size(1))
      << "next_tokens/embeddings seq len mismatch";
  if (num_speculative_tokens < 0) {
    num_speculative_tokens = static_cast<int32_t>(next_tokens.size(1) - 1);
  }
  CHECK_EQ(next_tokens.size(1),
           static_cast<int64_t>(num_speculative_tokens + 1))
      << "num_speculative_tokens mismatch";

  for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) {
    auto cur_tokens = next_tokens[i];
    auto cur_embeddings = embeddings[i];
    int32_t last_idx = -1;
    int32_t last_token_id = -1;
    for (int32_t j = 0; j < cur_tokens.size(0); ++j) {
      int64_t token = cur_tokens[j].item<int64_t>();
      if (token >= 0) {
        last_idx = j;
        last_token_id = static_cast<int32_t>(token);
      }
    }
    if (last_idx < 0) {
      continue;
    }

    set_last_state(ids[i], cur_embeddings[last_idx], last_token_id);
    auto& tail = mutable_tail(ids[i]);

    if (last_idx == num_speculative_tokens && num_speculative_tokens > 0) {
      const int32_t prev_idx = num_speculative_tokens - 1;
      int64_t prev_token = cur_tokens[prev_idx].item<int64_t>();
      if (prev_token >= 0) {
        tail.prev_embedding = cur_embeddings[prev_idx];
        tail.prev_token_id = static_cast<int32_t>(prev_token);
        tail.need_first_decode_fix = true;
      }
    }
  }
}

void EmbeddingCache::set_placeholder(const torch::Tensor& placeholder) {
  placeholder_ = placeholder;
}

// read embeddings from cache; empty slot returns placeholder if set (PD
// separation)
torch::Tensor EmbeddingCache::read(int32_t id) {
  const torch::Tensor& t = get_tail(id).embedding;
  if (t.defined()) {
    return t;
  }
  if (placeholder_.defined()) {
    return placeholder_.clone();
  }
  return t;
}

torch::Tensor EmbeddingCache::read(const std::vector<int32_t>& ids) {
  std::vector<torch::Tensor> tensors;
  int32_t total_nums = ids.size();
  tensors.reserve(total_nums);
  for (int32_t i = 0; i < total_nums; ++i) {
    tensors.emplace_back(read(ids[i]));
  }
  return torch::stack(tensors);
}

std::vector<EmbeddingCache::DecodeState> EmbeddingCache::read_for_decode(
    const std::vector<int32_t>& ids) {
  std::vector<DecodeState> items;
  items.reserve(ids.size());
  for (int32_t id : ids) {
    auto& tail = mutable_tail(id);
    DecodeState item = tail;
    if (!item.embedding.defined() && placeholder_.defined()) {
      item.embedding = placeholder_.clone();
    }
    // read_for_decode is consumptive for first-step-fix metadata.
    tail.need_first_decode_fix = false;
    tail.prev_embedding = torch::Tensor();
    tail.prev_token_id = -1;
    items.emplace_back(std::move(item));
  }
  return items;
}

void EmbeddingCache::clear_first_decode_fix(
    const std::vector<int32_t>& ids,
    const std::vector<uint8_t>& clear_mask) {
  CHECK_EQ(ids.size(), clear_mask.size()) << "clear mask size mismatch";
  for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) {
    if (clear_mask[i] == 0) {
      continue;
    }
    auto& tail = mutable_tail(ids[i]);
    tail.need_first_decode_fix = false;
    tail.prev_embedding = torch::Tensor();
    tail.prev_token_id = -1;
  }
}

EmbeddingCache::DecodeState& EmbeddingCache::mutable_tail(
    int32_t embedding_id) {
  CHECK_GE(embedding_id, 0);
  CHECK_LT(static_cast<size_t>(embedding_id), decode_tails_.size());
  return decode_tails_[embedding_id];
}

const EmbeddingCache::DecodeState& EmbeddingCache::get_tail(
    int32_t embedding_id) const {
  CHECK_GE(embedding_id, 0);
  CHECK_LT(static_cast<size_t>(embedding_id), decode_tails_.size());
  return decode_tails_[embedding_id];
}

void EmbeddingCache::set_last_state(int32_t embedding_id,
                                    const torch::Tensor& embedding,
                                    int32_t token_id) {
  auto& tail = mutable_tail(embedding_id);
  tail.embedding = embedding;
  tail.last_token_id = token_id;
  tail.prev_embedding = torch::Tensor();
  tail.prev_token_id = -1;
  tail.need_first_decode_fix = false;
}
}  // namespace xllm
