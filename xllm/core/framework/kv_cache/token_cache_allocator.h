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

#pragma once
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <vector>

#include "runtime/forward_params.h"

namespace xllm {

struct TokensCache {
  std::vector<int32_t> token_ids;

  TokensCache() = default;

  TokensCache(std::vector<int32_t>& ids) : token_ids(ids) {}
};

class TokenCacheAllocator final {
 public:
  TokenCacheAllocator(int32_t total_ids, int32_t num_speculative_tokens);

  ~TokenCacheAllocator();

  // disable copy, move and assign
  TokenCacheAllocator(const TokenCacheAllocator&) = delete;
  TokenCacheAllocator(TokenCacheAllocator&&) = delete;
  TokenCacheAllocator& operator=(const TokenCacheAllocator&) = delete;
  TokenCacheAllocator& operator=(TokenCacheAllocator&&) = delete;

  int32_t allocate();
  void free(int32_t eid);

  void write(int32_t eid, std::vector<int32_t> token_ids);
  void write(const std::vector<int32_t>& eids,
             std::vector<std::vector<int32_t>>& token_ids);
  void write(const std::vector<int32_t>& eids,
             std::vector<ForwardOutput>& outputs);

  std::vector<std::vector<int32_t>> read(const std::vector<int32_t>& eids);

  // get number of free ids
  size_t num_free_ids() const { return num_free_ids_; }

  // get number of total ids
  size_t num_total_ids() const { return free_ids_.size(); }

 private:
  // free ids count
  size_t num_free_ids_ = 0;

  // free ids list
  std::vector<int32_t> free_ids_;

  // tokens cache
  std::vector<TokensCache> tokens_cache_;

  int32_t num_speculative_tokens_;
};

}  // namespace xllm
