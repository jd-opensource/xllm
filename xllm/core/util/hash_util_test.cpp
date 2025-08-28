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

#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <string.h>

#include <iostream>

namespace xllm {

TEST(HashUtilTest, MurmurHash3) {
  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_EQ(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 5, 4};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {2, 1, 3, 5, 4};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {2, 1, 3, 5, 4};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmur_hash3(nullptr, tokens_1, hash_value_1);
    murmur_hash3(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }
}

}  // namespace xllm
