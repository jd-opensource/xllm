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

#include "prefix_cache.h"
#include "prefix_cache_hash.h"

namespace xllm {

class PrefixCacheHashSha256 final : public PrefixCacheHash {
 public:
  explicit PrefixCacheHashSha256(uint32_t block_size);

  ~PrefixCacheHashSha256();

  // match the token ids with the prefix tree
  // return matched blocks
  std::vector<Block> match(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {}) override;

  // insert the token ids and blocks into the prefix tree
  // return the length of new inserted tokens
  size_t insert(const Slice<int32_t>& token_ids,
                const Slice<Block>& blocks) override;

  // evict blocks hold by the prefix cache
  // return the actual number of evicted blocks
  size_t evict(size_t n_blocks) override;

  // get the number of blocks in the prefix cache
  size_t num_blocks() const override {
    CHECK(num_blocks_ == sha256_cached_blocks_.size())
        << "check block num failed";

    return num_blocks_;
  }

 private:
  std::unordered_map<Sha256Key,
                     Node*,
                     FixedStringKeyHash<Sha256Key>,
                     FixedStringKeyEqual<Sha256Key>>
      sha256_cached_blocks_;

  uint32_t hash_value_len_;
};

}  // namespace xllm
