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

#include <MurmurHash3.h>
#include <assert.h>
#include <gflags/gflags.h>

#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

DEFINE_uint32(murmur_hash3_seed, 1024, "default Murmur Hash seed");

namespace xllm {

void MurMurHash3::hash(const uint8_t* pre_hash_value,
                       const Slice<int32_t>& token_ids,
                       uint8_t* hash_value) {
  if (pre_hash_value == nullptr) {
    MurmurHash3_x64_128(reinterpret_cast<const void*>(token_ids.data()),
                        sizeof(int32_t) * token_ids.size(),
                        FLAGS_murmur_hash3_seed,
                        hash_value);
  } else {
    uint8_t key[1024];

    int32_t data_len =
        sizeof(int32_t) * token_ids.size() + MURMUR_HASH3_VALUE_LEN;
    assert(sizeof(key) > data_len);

    memcpy(key, pre_hash_value, MURMUR_HASH3_VALUE_LEN);
    memcpy(key + MURMUR_HASH3_VALUE_LEN,
           reinterpret_cast<const void*>(token_ids.data()),
           sizeof(int32_t) * token_ids.size());

    // print_hex_array(key, data_len);
    MurmurHash3_x64_128(reinterpret_cast<const void*>(key),
                        data_len,
                        FLAGS_murmur_hash3_seed,
                        hash_value);
  }
}

void print_hex_array(uint8_t* array, uint32_t len) {
  for (size_t i = 0; i < len; ++i) {
    unsigned char uc = static_cast<unsigned char>(array[i]);
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(uc);

    if (i % 16 == 15) {
      std::cout << std::endl;
    }

    else {
      std::cout << " ";
    }
  }
  std::cout << std::dec << std::endl;
}

}  // namespace xllm