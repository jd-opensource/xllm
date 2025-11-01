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

#include "hash_util.h"

#include <glog/logging.h>
#include <string.h>

#include "third_party/smhasher/src/MurmurHash3.h"

// Use a constant seed instead of FLAGS to avoid circular dependency
constexpr uint32_t MURMUR_HASH3_SEED = 0;

namespace xllm {

void murmur_hash3(const uint8_t* pre_hash_value,
                  const Slice<int32_t>& token_ids,
                  uint8_t* hash_value) {
  if (pre_hash_value == nullptr) {
    MurmurHash3_x64_128(reinterpret_cast<const void*>(token_ids.data()),
                        sizeof(int32_t) * token_ids.size(),
                        MURMUR_HASH3_SEED,
                        hash_value);
  } else {
    uint8_t key[1024];

    int32_t data_len =
        sizeof(int32_t) * token_ids.size() + MURMUR_HASH3_VALUE_LEN;
    CHECK_GT(sizeof(key), data_len) << "key size is too small";

    memcpy(key, pre_hash_value, MURMUR_HASH3_VALUE_LEN);
    memcpy(key + MURMUR_HASH3_VALUE_LEN,
           reinterpret_cast<const void*>(token_ids.data()),
           sizeof(int32_t) * token_ids.size());

    MurmurHash3_x64_128(reinterpret_cast<const void*>(key),
                        data_len,
                        MURMUR_HASH3_SEED,
                        hash_value);
  }
}

}  // namespace xllm
