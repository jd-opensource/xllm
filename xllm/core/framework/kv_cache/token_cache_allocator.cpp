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

#include "token_cache_allocator.h"

#include <glog/logging.h>

#include <cstdint>
#include <vector>

namespace xllm {

TokenCacheAllocator::TokenCacheAllocator(int32_t total_ids,
                                         int32_t num_speculative_tokens)
    : num_free_ids_(total_ids),
      num_speculative_tokens_(num_speculative_tokens) {
  CHECK_GT(total_ids, 0) << "No ids to allocate";

  tokens_cache_.resize(total_ids);
  free_ids_.reserve(total_ids);
  for (int32_t i = 0; i < total_ids; ++i) {
    free_ids_.push_back(total_ids - i - 1);
  }
}

TokenCacheAllocator::~TokenCacheAllocator() {
  CHECK(num_free_ids_ == free_ids_.size()) << "Not all ids have been freed";
}

// allocate a id
int32_t TokenCacheAllocator::allocate() {
  CHECK(num_free_ids_ > 0) << "No more ids available";
  const int32_t eid = free_ids_[--num_free_ids_];
  return eid;
}

// caller should make sure the eid is valid
void TokenCacheAllocator::free(int32_t eid) {
  CHECK(num_free_ids_ < free_ids_.size());
  free_ids_[num_free_ids_++] = eid;
}

// write token ids to cache
void TokenCacheAllocator::write(int32_t eid, std::vector<int32_t> token_ids) {
  while (token_ids.size() < num_speculative_tokens_) {
    token_ids.push_back(0);
  }
  tokens_cache_[eid].token_ids = std::move(token_ids);
}

void TokenCacheAllocator::write(const std::vector<int32_t>& eids,
                                std::vector<std::vector<int32_t>>& token_ids) {
  int32_t num_sequences = eids.size();
  for (int32_t i = 0; i < num_sequences; ++i) {
    write(eids[i], token_ids[i]);
  }
}

void TokenCacheAllocator::write(const std::vector<int32_t>& eids,
                                std::vector<ForwardOutput>& outputs) {
  int32_t num_sequences = eids.size();
  num_sequences = outputs[0].sample_output.next_tokens.size(0);
  std::vector<torch::Tensor> next_tokens_vec;
  for (auto& output : outputs) {
    auto next_tokens =
        output.sample_output.next_tokens.view({num_sequences, 1});
    next_tokens_vec.push_back(next_tokens);
  }

  // concatenate the next tokens along the last dimension
  // [batch_size, num_speculative_tokens]
  auto next_tokens = torch::cat(next_tokens_vec, /*dim=*/1);
  next_tokens = safe_to(next_tokens, torch::kInt32);
  next_tokens = safe_to(next_tokens, torch::kCPU);
  for (int32_t i = 0; i < num_sequences; ++i) {
    auto token_ids = next_tokens[i];
    Slice<int32_t> token_ids_slice(token_ids.data_ptr<int32_t>(),
                                   token_ids.size(0));
    write(eids[i], token_ids_slice);
  }
}

std::vector<std::vector<int32_t>> TokenCacheAllocator::read(
    const std::vector<int32_t>& eids) {
  std::vector<std::vector<int32_t>> token_ids;
  int32_t num_sequences = eids.size();
  token_ids.reserve(num_sequences);
  for (int32_t i = 0; i < num_sequences; ++i) {
    token_ids.emplace_back(tokens_cache_[eids[i]].token_ids);
  }
  return token_ids;
}

}  // namespace xllm
