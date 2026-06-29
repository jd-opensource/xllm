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

#include <sstream>
#include <unordered_map>

namespace xllm::kernel::mlu {
template <typename Key>
int32_t lookup_algo_id(const std::unordered_map<Key, int32_t>& map,
                       const Key& key,
                       const char* dim_name) {
  auto it = map.find(key);
  if (it == map.end()) {
    std::ostringstream supported;
    for (const auto& [k, v] : map) {
      supported << k << " ";
    }
    LOG(FATAL) << "Unsupported " << dim_name << " " << key
               << " for tmo_kernel; no pre-compiled kernel variant is "
               << "available. Supported values: " << supported.str();
  }
  return it->second;
}

}  // namespace xllm::kernel::mlu
