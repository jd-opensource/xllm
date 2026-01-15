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

#include "embedding_cache.h"

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace xllm {

EmbeddingCache::EmbeddingCache(int32_t total_nums) {
  CHECK_GT(total_nums, 0) << "No embeddings to allocate";
  cache_.resize(total_nums);
}

// write embeddings to cache
void EmbeddingCache::write(int32_t id, const torch::Tensor& embeddings) {
  cache_[id] = embeddings;
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
                                    torch::Tensor& next_tokens,
                                    const torch::Tensor& embeddings) {
  int32_t total_nums = ids.size();
  for (int32_t i = 0; i < total_nums; ++i) {
    torch::Tensor cur_tokens = next_tokens[i];
    for (int32_t j = 0; j < cur_tokens.size(0); ++j) {
      if (cur_tokens[j].item<int32_t>() >= 0) {
        write(ids[i], embeddings[i][j]);
      }
    }
  }
}

// read embeddings from cache
torch::Tensor EmbeddingCache::read(int32_t id) { return cache_[id]; }

torch::Tensor EmbeddingCache::read(const std::vector<int32_t>& ids) {
  std::vector<torch::Tensor> tensors;
  int32_t total_nums = ids.size();
  tensors.reserve(total_nums);
  for (int32_t i = 0; i < total_nums; ++i) {
    tensors.emplace_back(cache_[ids[i]]);
  }
  return torch::stack(tensors);
}
}  // namespace xllm
