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

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_input.h"
#include "core/util/hash_util.h"

namespace xllm {

class PromptProcessor {
 public:
  virtual ~PromptProcessor() = default;

  virtual void process(std::string& prompt, const MMData& mm_data) = 0;
  void hash_mm_items(MMInput& mm_input, MMData& mm_data) {
    const auto& mm_input_items = mm_input.items();
    auto& mm_items = mm_data.items<MMItemVec>();
    size_t size = mm_input_items.size();
    for (size_t idx = 0; idx < size; ++idx) {
      const std::string& data = mm_input_items[idx].raw_data;
      if (!data.empty()) {
        XXH3Key mm_hash = hash_string(data);
        auto& schedule_data =
            mm_items[idx].mutable_state().mutable_schedule_data();
        schedule_data.key = mm_hash;
      }
    }
  }
  virtual void find_mm_spans(const std::vector<int32_t>& token_ids,
                             MMData& mm_data) {};
};

}  // namespace xllm
