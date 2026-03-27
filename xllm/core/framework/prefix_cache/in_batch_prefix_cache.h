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

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "framework/request/sequence.h"
#include "util/hash_util.h"

namespace xllm {

class InBatchPrefixCacheContext final {
 public:
  void try_match(Sequence* sequence, size_t block_size);
  void register_provider(Sequence* sequence,
                         size_t block_size,
                         size_t max_handle_num_tokens);

 private:
  struct Provider {
    Sequence* sequence = nullptr;
    int32_t dp_rank = -1;
    size_t available_full_blocks = 0;
  };

  struct LookupKey {
    size_t block_index = 0;
    XXH3Key hash;
  };

  struct LookupKeyHash {
    size_t operator()(const LookupKey& key) const;
  };

  struct LookupKeyEqual {
    bool operator()(const LookupKey& lhs, const LookupKey& rhs) const;
  };

  std::vector<Provider> providers_;
  std::unordered_map<LookupKey,
                     std::vector<size_t>,
                     LookupKeyHash,
                     LookupKeyEqual>
      provider_index_;
};

}  // namespace xllm
