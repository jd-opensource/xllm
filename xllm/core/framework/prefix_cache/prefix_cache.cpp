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

#include "prefix_cache.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include "prefix_cache_hash_murmur3.h"
#include "prefix_cache_hash_sha256.h"

namespace xllm {

std::unique_ptr<PrefixCache> CreatePrefixCachePolicy(
    int32_t block_size,
    const std::string& policy,
    const bool& enbale_service_routing) {
  std::vector<absl::string_view> subs = absl::StrSplit(policy, ':');
  CHECK(subs.size() > 0) << " Prefix cache, input param invalid."
                         << " policy:" << policy;

  if ("sha256_hash" == subs[0]) {
    return std::make_unique<PrefixCacheHashSha256>(block_size);
  } else if ("murmur_hash3" == subs[0]) {
    return std::make_unique<PrefixCacheHashMurmur3>(block_size,
                                                    enbale_service_routing);
  } else {
    return nullptr;
  }
}

}  // namespace xllm
