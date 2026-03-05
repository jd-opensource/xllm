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

#include "util/utils.h"

namespace xllm {

EmbeddingCache::EmbeddingCache(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No embeddings to allocate";
  decode_tails_.resize(total_nums);
}

void EmbeddingCache::write(const std::vector<int32_t>& ids,
                           const torch::Tensor& next_tokens,
                           const torch::Tensor& embeddings,
                           const torch::Tensor& probs,
                           const torch::Tensor& accepted_tokens) {
  torch::Tensor accepted_tokens_cpu = safe_to(accepted_tokens, torch::kCPU);
  const bool has_accepted_tokens = accepted_tokens.defined();

  for (int32_t i = 0; i < static_cast<int32_t>(ids.size()); ++i) {
    auto& tail = mutable_tail(ids[i]);
    tail.token_id = static_cast<int32_t>(next_tokens[i].item<int64_t>());
    tail.correction_token_id = tail.token_id;
    tail.correction_position_offset = 0;
    if (has_accepted_tokens) {
      if (accepted_tokens_cpu.dim() == 1) {
        int64_t token = accepted_tokens_cpu[i].item<int64_t>();
        tail.correction_token_id = static_cast<int32_t>(token);
      } else {
        int32_t last_valid_token = -1;
        int32_t last_valid_idx = -1;
        const int32_t token_width = accepted_tokens_cpu.size(1);
        for (int32_t j = 0; j < token_width; ++j) {
          int64_t token = accepted_tokens_cpu[i][j].item<int64_t>();
          if (token >= 0) {
            last_valid_token = static_cast<int32_t>(token);
            last_valid_idx = j;
          }
        }
        tail.correction_token_id = last_valid_token;
        tail.correction_position_offset = last_valid_idx;
      }
    }
    tail.embedding = embeddings[i];
    tail.probs = probs[i];
  }
}

void EmbeddingCache::set_placeholder(
    const torch::Tensor& embedding_placeholder) {
  embedding_placeholder_ = embedding_placeholder;
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

std::vector<int32_t> EmbeddingCache::read_correction_tokens(
    const std::vector<int32_t>& ids) const {
  CHECK(!ids.empty()) << "decode ids should not be empty";
  std::vector<int32_t> tokens;
  tokens.reserve(ids.size());
  for (int32_t id : ids) {
    const auto& item = get_tail(id);
    CHECK_GE(item.correction_token_id, 0)
        << "decode entry missing correction token id";
    tokens.emplace_back(item.correction_token_id);
  }
  return tokens;
}

std::vector<int32_t> EmbeddingCache::read_position_offsets(
    const std::vector<int32_t>& ids) const {
  CHECK(!ids.empty()) << "decode ids should not be empty";
  std::vector<int32_t> offsets;
  offsets.reserve(ids.size());
  for (int32_t id : ids) {
    const auto& item = get_tail(id);
    CHECK_GE(item.correction_token_id, 0)
        << "decode entry missing correction token id";
    offsets.emplace_back(item.correction_position_offset);
  }
  return offsets;
}

void EmbeddingCache::clear(const std::vector<int32_t>& ids) {
  for (int32_t id : ids) {
    auto& tail = mutable_tail(id);
    tail.embedding = torch::Tensor();
    tail.token_id = -1;
    tail.correction_token_id = -1;
    tail.correction_position_offset = 0;
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
