/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include "prefix_cache_hash.h"

#include <gtest/gtest.h>

#include "framework/block/block_manager_impl.h"

namespace xllm {

void test_basic_operation(BlockManagerImpl* block_manager,
                          PrefixCacheHash* prefix_cache_hash,
                          uint32_t block_size) {
  EXPECT_EQ(prefix_cache_hash->num_blocks(), 0);

  // token_ids number must be greater than  2 * block_size here
  std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Slice<int32_t> slice_token_ids(token_ids);
  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);

    EXPECT_EQ(block_matched.size(), 0);
  }

  uint32_t n_blocks = token_ids.size() / block_size;
  {
    std::vector<Block> token_blocks = block_manager->allocate(n_blocks);
    Slice<Block> slice_token_blocks(token_blocks);

    prefix_cache_hash->insert(slice_token_ids, slice_token_blocks);
  }

  EXPECT_EQ(prefix_cache_hash->num_blocks(), n_blocks);

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  {
    auto block_matched = prefix_cache_hash->match(
        slice_token_ids.slice(block_size, 2 * block_size));
    EXPECT_EQ(block_matched.size(), 0);
  }

  EXPECT_EQ(prefix_cache_hash->evict(1), 1);

  EXPECT_EQ(prefix_cache_hash->num_blocks(), n_blocks - 1);

  {
    auto block_matched =
        prefix_cache_hash->match(slice_token_ids.slice(0, block_size));
    EXPECT_EQ(block_matched.size(), 1);
  }

  {
    auto block_matched =
        prefix_cache_hash->match(slice_token_ids.slice(block_size));
    EXPECT_EQ(block_matched.size(), 0);
  }
}

TEST(PrefixCacheHashTest, BasicOperation) {
  const uint32_t block_size = 4;
  const uint32_t total_blocks = 5;
  BlockManager::Options options;
  options.num_blocks(total_blocks).block_size(block_size);
  BlockManagerImpl block_manager(options);

  PrefixCacheHash prefix_cache_hash(block_size);

  test_basic_operation(&block_manager, &prefix_cache_hash, block_size);
}

void test_insert_operation(BlockManagerImpl* block_manager,
                           PrefixCacheHash* prefix_cache_hash,
                           uint32_t block_size) {
  EXPECT_EQ(prefix_cache_hash->num_blocks(), 0);

  // insert two-block firstly
  // token_ids number must be greater than  2 * block_size here
  std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Slice<int32_t> slice_token_ids(token_ids);

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);

    EXPECT_EQ(block_matched.size(), 0);
  }

  uint32_t n_blocks = token_ids.size() / block_size;

  {
    std::vector<Block> token_blocks = block_manager->allocate(n_blocks);
    Slice<Block> slice_token_blocks(token_blocks);

    prefix_cache_hash->insert(slice_token_ids, slice_token_blocks);

    EXPECT_EQ(prefix_cache_hash->num_blocks(), n_blocks);
  }

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  // insert another two-block
  std::vector<int32_t> token_ids_1 = {9, 10, 11, 12, 13, 14, 15, 16, 17};
  Slice<int32_t> slice_token_ids_1(token_ids_1);

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids_1);

    EXPECT_EQ(block_matched.size(), 0);
  }

  n_blocks = token_ids_1.size() / block_size;

  {
    std::vector<Block> token_blocks_1 = block_manager->allocate(n_blocks);
    Slice<Block> slice_token_blocks_1(token_blocks_1);

    prefix_cache_hash->insert(slice_token_ids_1, slice_token_blocks_1);

    EXPECT_EQ(prefix_cache_hash->num_blocks(), 2 * n_blocks);
  }

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids_1);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  EXPECT_EQ(prefix_cache_hash->evict(1), 1);
  EXPECT_EQ(prefix_cache_hash->num_blocks(), 2 * n_blocks - 1);

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids_1);
    EXPECT_EQ(block_matched.size(), n_blocks - 1);
  }

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  EXPECT_EQ(prefix_cache_hash->evict(1), 1);
  EXPECT_EQ(prefix_cache_hash->num_blocks(), 2 * n_blocks - 2);

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids_1);
    EXPECT_EQ(block_matched.size(), 0);
  }

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);

    Slice<Block> slice_token_blocks(block_matched);
    prefix_cache_hash->insert(slice_token_ids, slice_token_blocks);
  }

  EXPECT_EQ(prefix_cache_hash->num_blocks(), n_blocks);

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  prefix_cache_hash->evict(1);
  EXPECT_EQ(prefix_cache_hash->num_blocks(), n_blocks - 1);

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids_1);
    EXPECT_EQ(block_matched.size(), 0);
  }

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks - 1);
  }
}

TEST(PrefixCacheHashTest, InsertOperation) {
  const uint32_t block_size = 4;
  const uint32_t total_blocks = 5;
  BlockManager::Options options;
  options.num_blocks(total_blocks).block_size(block_size);
  BlockManagerImpl block_manager(options);

  PrefixCacheHash prefix_cache_hash(block_size);

  test_insert_operation(&block_manager, &prefix_cache_hash, block_size);
}

void test_evict_operation(BlockManagerImpl* block_manager,
                          PrefixCacheHash* prefix_cache_hash,
                          uint32_t block_size) {
  EXPECT_EQ(prefix_cache_hash->num_blocks(), 0);

  prefix_cache_hash->evict(1);
  EXPECT_EQ(prefix_cache_hash->num_blocks(), 0);

  // insert two-block firstly
  // token_ids number must be greater than  2 * block_size here
  std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Slice<int32_t> slice_token_ids(token_ids);

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);

    EXPECT_EQ(block_matched.size(), 0);
  }

  uint32_t n_blocks = token_ids.size() / block_size;

  {
    std::vector<Block> token_blocks = block_manager->allocate(n_blocks);
    Slice<Block> slice_token_blocks(token_blocks);

    prefix_cache_hash->insert(slice_token_ids, slice_token_blocks);

    EXPECT_EQ(prefix_cache_hash->num_blocks(), n_blocks);
  }

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  EXPECT_EQ(block_manager->num_free_blocks(),
            block_manager->num_total_blocks() - n_blocks);

  EXPECT_EQ(prefix_cache_hash->evict(n_blocks), n_blocks);

  EXPECT_EQ(block_manager->num_free_blocks(),
            block_manager->num_total_blocks());

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), 0);
  }

  {
    std::vector<Block> token_blocks = block_manager->allocate(n_blocks);
    Slice<Block> slice_token_blocks(token_blocks);
    prefix_cache_hash->insert(slice_token_ids, slice_token_blocks);

    EXPECT_EQ(prefix_cache_hash->num_blocks(), n_blocks);
  }

  {
    auto block_matched = prefix_cache_hash->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }
}

TEST(PrefixCacheHashTest, EvictOperation) {
  const uint32_t block_size = 4;
  const uint32_t total_blocks = 5;
  BlockManager::Options options;
  options.num_blocks(total_blocks).block_size(block_size);
  BlockManagerImpl block_manager(options);

  PrefixCacheHash prefix_cache_hash(block_size);

  test_evict_operation(&block_manager, &prefix_cache_hash, block_size);
}

}  // namespace xllm
