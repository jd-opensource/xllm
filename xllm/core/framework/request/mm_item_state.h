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

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "core/util/hash_util.h"

namespace xllm {

struct MMItemState {
  struct TokenPos {
    uint32_t offset = 0;
    uint32_t length = 0;
  };

  struct PrefixCache {
    Murmur3Key key;
    uint32_t cached_token_num = 0;
  };

  const TokenPos& token_pos() const { return token_pos_; }
  TokenPos& mutable_token_pos() { return token_pos_; }

  const PrefixCache& prefix_cache() const { return prefix_cache_; }
  PrefixCache& mutable_prefix_cache() { return prefix_cache_; }

  bool prefix_cached() const;
  bool prefix_complete_cached() const;

 private:
  TokenPos token_pos_;
  PrefixCache prefix_cache_;
};

}  // namespace xllm
