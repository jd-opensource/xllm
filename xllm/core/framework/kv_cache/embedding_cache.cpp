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
#include <vector>

namespace xllm {

EmbeddingCache::EmbeddingCache(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No embeddings to allocate";
  decode_tails_.resize(total_nums);
}

void EmbeddingCache::write(const std::vector<int32_t>& ids,
                           const torch::Tensor& next_tokens,
                           const torch::Tensor& embeddings,
                           const torch::Tensor& probs) {
  CHECK_EQ(next_tokens.dim(), 1) << "next_tokens should be [batch]";
  CHECK_EQ(embeddings.dim(), 2) << "embeddings should be [batch, h]";
  CHECK_EQ(next_tokens.size(0), static_cast<int64_t>(ids.size()))
      << "next_tokens batch mismatch";
  CHECK_EQ(embeddings.size(0), static_cast<int64_t>(ids.size()))
      << "embeddings batch mismatch";
  CHECK(probs.defined()) << "probs should be defined";
  CHECK_GE(probs.dim(), 1) << "probs should have batch dimension";
  CHECK_EQ(probs.size(0), static_cast<int64_t>(ids.size()))
      << "probs batch mismatch";

  for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) {
    auto& tail = mutable_tail(ids[i]);
    tail.token_id = static_cast<int32_t>(next_tokens[i].item<int64_t>());
    tail.embedding = embeddings[i];
    tail.probs = probs[i];
  }
}

ForwardOutput EmbeddingCache::read_for_decode(const std::vector<int32_t>& ids) {
  CHECK(!ids.empty()) << "decode ids should not be empty";
  std::vector<int32_t> token_ids;
  std::vector<torch::Tensor> embeddings;
  std::vector<torch::Tensor> probs;
  token_ids.reserve(ids.size());
  embeddings.reserve(ids.size());
  probs.reserve(ids.size());
  for (int32_t id : ids) {
    const auto& item = get_tail(id);
    CHECK_GE(item.token_id, 0) << "decode entry missing token id";
    CHECK(item.embedding.defined()) << "decode entry missing embedding";
    CHECK(item.probs.defined()) << "decode entry missing probs";
    token_ids.emplace_back(item.token_id);
    embeddings.emplace_back(item.embedding);
    probs.emplace_back(item.probs);
  }
  ForwardOutput output;
  output.sample_output.next_tokens = torch::tensor(token_ids, torch::kInt);
  output.sample_output.embeddings = torch::stack(embeddings);
  output.sample_output.probs = torch::stack(probs);
  return output;
}

void EmbeddingCache::clear(const std::vector<int32_t>& ids) {
  for (int32_t id : ids) {
    auto& tail = mutable_tail(id);
    tail.embedding = torch::Tensor();
    tail.token_id = -1;
    tail.probs = torch::Tensor();
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
}  // namespace xllm
