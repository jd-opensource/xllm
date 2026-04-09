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

#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/request/request_output.h"

namespace xllm {

inline std::vector<int64_t> select_rec_item_ids(const SequenceOutput& output) {
  if (!FLAGS_enable_rec_multi_item_output || output.item_ids_list.empty()) {
    if (output.item_ids.has_value()) {
      return {output.item_ids.value()};
    }
    return {};
  }

  std::vector<int64_t> selected_item_ids;
  selected_item_ids.reserve(output.item_ids_list.size());
  std::unordered_set<int64_t> seen_item_ids;
  for (const int64_t item_id : output.item_ids_list) {
    if (seen_item_ids.insert(item_id).second) {
      selected_item_ids.emplace_back(item_id);
    }
  }

  const int32_t each_threshold = FLAGS_each_conversion_threshold;
  if (each_threshold > 0 &&
      static_cast<int32_t>(selected_item_ids.size()) > each_threshold) {
    uint32_t seed = FLAGS_random_seed >= 0
                        ? static_cast<uint32_t>(FLAGS_random_seed) +
                              static_cast<uint32_t>(output.index)
                        : std::random_device{}();
    std::mt19937 generator(seed);
    std::shuffle(selected_item_ids.begin(), selected_item_ids.end(), generator);
    selected_item_ids.resize(each_threshold);
  }

  return selected_item_ids;
}

}  // namespace xllm
