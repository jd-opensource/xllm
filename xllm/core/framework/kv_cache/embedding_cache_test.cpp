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

#include "embedding_cache.h"

#include <gtest/gtest.h>

namespace xllm {

namespace {

bool tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  return lhs.defined() && rhs.defined() && torch::equal(lhs, rhs);
}

}  // namespace

TEST(EmbeddingCacheTest, WriteValidateAllAcceptedSetsFirstDecodeFix) {
  EmbeddingCache cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {1};
  auto next_tokens = torch::tensor({{11, 12, 13, 14}}, torch::kInt);
  auto embeddings =
      torch::tensor({{{1.0f, 1.1f}, {2.0f, 2.1f}, {3.0f, 3.1f}, {4.0f, 4.1f}}},
                    torch::kFloat);

  cache.write_validate(
      ids, next_tokens, embeddings, /*num_speculative_tokens=*/3);

  auto items = cache.read_for_decode(ids);
  ASSERT_EQ(items.size(), 1);
  const auto& item = items[0];
  EXPECT_TRUE(item.need_first_decode_fix);
  EXPECT_EQ(item.prev_token_id, 13);
  EXPECT_EQ(item.last_token_id, 14);
  EXPECT_TRUE(tensor_equal(item.prev_embedding, embeddings[0][2]));
  EXPECT_TRUE(tensor_equal(item.embedding, embeddings[0][3]));

  // read_for_decode() should consume first-decode-fix state.
  auto items_after_clear = cache.read_for_decode(ids);
  ASSERT_EQ(items_after_clear.size(), 1);
  EXPECT_FALSE(items_after_clear[0].need_first_decode_fix);
  EXPECT_EQ(items_after_clear[0].prev_token_id, -1);
  EXPECT_FALSE(items_after_clear[0].prev_embedding.defined());
}

TEST(EmbeddingCacheTest, WriteValidatePartialAcceptedNoFirstDecodeFix) {
  EmbeddingCache cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {2};
  auto next_tokens = torch::tensor({{21, 22, -1, -1}}, torch::kInt);
  auto embeddings = torch::tensor(
      {{{10.0f, 10.1f}, {20.0f, 20.1f}, {30.0f, 30.1f}, {40.0f, 40.1f}}},
      torch::kFloat);

  cache.write_validate(
      ids, next_tokens, embeddings, /*num_speculative_tokens=*/3);

  auto items = cache.read_for_decode(ids);
  ASSERT_EQ(items.size(), 1);
  const auto& item = items[0];
  EXPECT_FALSE(item.need_first_decode_fix);
  EXPECT_EQ(item.prev_token_id, -1);
  EXPECT_EQ(item.last_token_id, 22);
  EXPECT_FALSE(item.prev_embedding.defined());
  EXPECT_TRUE(tensor_equal(item.embedding, embeddings[0][1]));
}

TEST(EmbeddingCacheTest, LegacyWriteValidateCompatibleWithDecodeTailState) {
  EmbeddingCache cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {3};
  auto next_tokens = torch::tensor({{31, 32, 33}}, torch::kInt);
  auto embeddings = torch::tensor({{{1.0f}, {2.0f}, {3.0f}}}, torch::kFloat);

  cache.write_validate(ids, next_tokens, embeddings);

  auto items = cache.read_for_decode(ids);
  ASSERT_EQ(items.size(), 1);
  EXPECT_TRUE(items[0].need_first_decode_fix);
  EXPECT_EQ(items[0].prev_token_id, 32);
  EXPECT_EQ(items[0].last_token_id, 33);
  EXPECT_TRUE(tensor_equal(items[0].embedding, embeddings[0][2]));
}

}  // namespace xllm
