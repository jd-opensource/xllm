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

#include <string>

#include "core/framework/request/mm_data.h"
#include "framework/request/mm_input.h"

namespace xllm {

class InputProcessor {
 public:
  virtual ~InputProcessor() = default;

  virtual void process(std::string& prompt, const MMData& mm_data) = 0;
  virtual void find_mm_spans(const std::vector<int>& prompt, MMData& mm_data) {
  };
  void hash_mm_items(MMInput& mm_input, MMData& mm_data) {
    std::vector<Murmur3Key> mm_hashes;
    const auto& mm_input_items = mm_input.items();
    auto& mm_items = mm_data.items<MMItemVec>();
    mm_hashes.reserve(mm_input.size());
    int32_t mm_input_size = mm_input.size();
    for (int32_t idx = 0; idx < mm_input_size; ++idx) {
      auto data = mm_input_items[idx].raw_data;
      if (!data.empty()) {
        auto mm_hash = hash_string(data);
        auto& prefix_cache =
            mm_items[idx].mutable_state().mutable_prefix_cache();
        prefix_cache.key = mm_hash;
      }
    }
  }
};

}  // namespace xllm
