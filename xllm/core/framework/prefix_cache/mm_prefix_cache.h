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

#pragma once

#include <glog/logging.h>

#include <vector>

#include "prefix_cache.h"
#include "util/double_buffer.h"

namespace xllm {
class Sequence;
class MMData;

void mm_murmur_hash3(const std::vector<const uint8_t*>& mm_hash_values,
                     const uint8_t* pre_hash_value,
                     const Slice<int32_t>& token_ids,
                     uint8_t* hash_value);

class MMPrefixCache final : public PrefixCache {
 public:
  explicit MMPrefixCache(uint32_t block_size);

  ~MMPrefixCache();

  std::vector<Block> match(
      Sequence* sequence,
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {}) override;

 protected:
  size_t insert(Sequence* sequence,
                const Slice<int32_t>& token_ids,
                std::vector<Block>& blocks,
                size_t existed_shared_blocks_num,
                std::vector<Murmur3Key>* insert_keys) override;
  std::vector<const uint8_t*> get_block_mm_hash_values(const MMData& mm_data,
                                                       int32_t start_token_idx,
                                                       int32_t end_token_idx,
                                                       int32_t& start_mm_idx);

 private:
  ThreadPool threadpool_;
};

}  // namespace xllm
