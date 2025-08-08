
#include "hash_util.h"

#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <string.h>

#include <iostream>

namespace xllm {

TEST(HashUtilTest, MurmurHash3) {
  MurMurHash3 murmurhash3;
  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[MURMUR_HASH3_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5};
    uint8_t hash_value_2[MURMUR_HASH3_VALUE_LEN];

    murmurhash3.hash(nullptr, tokens_1, hash_value_1);
    murmurhash3.hash(nullptr, tokens_2, hash_value_2);

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

    murmurhash3.hash(nullptr, tokens_1, hash_value_1);
    murmurhash3.hash(nullptr, tokens_2, hash_value_2);

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

    murmurhash3.hash(nullptr, tokens_1, hash_value_1);
    murmurhash3.hash(nullptr, tokens_2, hash_value_2);

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

    murmurhash3.hash(nullptr, tokens_1, hash_value_1);
    murmurhash3.hash(nullptr, tokens_2, hash_value_2);

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

    murmurhash3.hash(nullptr, tokens_1, hash_value_1);
    murmurhash3.hash(nullptr, tokens_2, hash_value_2);

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

    murmurhash3.hash(nullptr, tokens_1, hash_value_1);
    murmurhash3.hash(nullptr, tokens_2, hash_value_2);

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

    murmurhash3.hash(nullptr, tokens_1, hash_value_1);
    murmurhash3.hash(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(strncmp(reinterpret_cast<const char*>(hash_value_1),
                      reinterpret_cast<const char*>(hash_value_2),
                      MURMUR_HASH3_VALUE_LEN),
              0);
  }
}

}  // namespace xllm
