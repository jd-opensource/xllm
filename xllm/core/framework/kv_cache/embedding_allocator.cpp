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

#include "embedding_allocator.h"

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace xllm {

EmbeddingAllocator::EmbeddingAllocator(int32_t total_embeddings,
                                       int32_t embedding_dim,
                                       torch::ScalarType dtype)
    : num_free_embeddings_(total_embeddings) {
  CHECK_GT(total_embeddings, 0) << "No embeddings to allocate";

  embeddings_cache_tensor_ =
      torch::zeros({total_embeddings, embedding_dim}, dtype);
  embeddings_cache_ = torch::unbind(embeddings_cache_tensor_, /*dim*/ 0);
  free_embeddings_.reserve(total_embeddings);
  for (int32_t i = 0; i < total_embeddings; ++i) {
    free_embeddings_.push_back(total_embeddings - i - 1);
  }
}

EmbeddingAllocator::~EmbeddingAllocator() {
  CHECK(num_free_embeddings_ == free_embeddings_.size())
      << "Not all embeddings have been freed";
}

// allocate a embedding id
int32_t EmbeddingAllocator::allocate() {
  CHECK(num_free_embeddings_ > 0) << "No more embeddings available";
  const int32_t embedding_id = free_embeddings_[--num_free_embeddings_];
  return embedding_id;
}

// caller should make sure the embedding_id is valid
void EmbeddingAllocator::free(int32_t embedding_id) {
  CHECK(num_free_embeddings_ < free_embeddings_.size());
  free_embeddings_[num_free_embeddings_++] = embedding_id;
}

// write embeddings to cache
void EmbeddingAllocator::write(int32_t embedding_id,
                               const torch::Tensor& embeddings) {
  // embeddings_cache_[embedding_id] = embeddings;
  // aclrtSynchronizeDevice();
  embeddings_cache_[embedding_id].copy_(embeddings);
}

void EmbeddingAllocator::write(const std::vector<int32_t>& embedding_ids,
                               const torch::Tensor& embeddings) {
  int32_t total_embeddings = embedding_ids.size();
  CHECK_EQ(total_embeddings, embeddings.size(0));
  for (int32_t i = 0; i < total_embeddings; ++i) {
    write(embedding_ids[i], embeddings[i]);
  }
}

void EmbeddingAllocator::write_validate(
    const std::vector<int32_t>& embedding_ids,
    torch::Tensor&& next_tokens,
    const torch::Tensor& embeddings) {
  int32_t num_sequences = embedding_ids.size();
  for (int32_t i = 0; i < num_sequences; ++i) {
    torch::Tensor cur_tokens = next_tokens[i];
    for (int32_t j = 0; j < cur_tokens.size(0); ++j) {
      if (cur_tokens[j].item<int32_t>() >= 0) {
        write(embedding_ids[i], embeddings[i][j]);
      }
    }
  }
}

// read embeddings from cache
torch::Tensor EmbeddingAllocator::read(int32_t embedding_id) {
  return embeddings_cache_[embedding_id];
}

torch::Tensor EmbeddingAllocator::read(
    const std::vector<int32_t>& embedding_ids) {
  std::vector<torch::Tensor> embeddings;
  int32_t total_embeddings = embedding_ids.size();
  embeddings.reserve(total_embeddings);
  for (int32_t i = 0; i < total_embeddings; ++i) {
    embeddings.emplace_back(embeddings_cache_[embedding_ids[i]]);
  }
  return torch::stack(embeddings);
}
}  // namespace xllm
