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

#include "mm_batch_data.h"

#include <gtest/gtest.h>

#include <cstring>

namespace xllm {
namespace {

void fill_prefix_cache_key(MMItemState& state) {
  for (uint32_t i = 0; i < XXH3_128BITS_HASH_VALUE_LEN; ++i) {
    state.mutable_prefix_cache().key.data[i] = static_cast<uint8_t>(i + 1);
  }
}

void expect_tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  ASSERT_TRUE(lhs.defined());
  ASSERT_TRUE(rhs.defined());
  EXPECT_TRUE(torch::equal(lhs, rhs));
}

TEST(MMBatchDataProtoTest, RoundTripsItemVecAndState) {
  MMDict item_dict;
  item_dict.emplace("pixel_values",
                    torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                  torch::kFloat32));
  item_dict.emplace("image_grid_thw", torch::tensor({1, 2, 3}, torch::kInt32));
  item_dict.emplace(
      "aux",
      std::vector<torch::Tensor>{torch::tensor({5, 6}, torch::kInt64),
                                 torch::tensor({7, 8}, torch::kInt32)});

  MMDataItem item(MMType::IMAGE, item_dict);
  item.mutable_state().mutable_token_pos().offset = 7;
  item.mutable_state().mutable_token_pos().length = 9;
  item.mutable_state().mutable_prefix_cache().cached_token_num = 3;
  fill_prefix_cache_key(item.mutable_state());

  MMData mm_data(MMType::IMAGE, MMItemVec{item});
  MMBatchData batch(std::vector<MMData>{mm_data});

  proto::MMData pb_mm_data;
  ASSERT_TRUE(mmdata_to_proto(batch, &pb_mm_data));
  ASSERT_EQ(pb_mm_data.entries_size(), 1);
  EXPECT_TRUE(pb_mm_data.entries(0).is_item_vec());
  ASSERT_EQ(pb_mm_data.entries(0).items_size(), 1);

  MMBatchData roundtrip;
  ASSERT_TRUE(proto_to_mmdata(pb_mm_data, &roundtrip));
  ASSERT_EQ(roundtrip.mm_data_vec().size(), 1);
  ASSERT_TRUE(roundtrip.mm_data_vec()[0].hold<MMItemVec>());
  ASSERT_EQ(roundtrip.mm_data_vec()[0].items<MMItemVec>().size(), 1);
  ASSERT_TRUE(roundtrip.has("pixel_values"));
  ASSERT_TRUE(roundtrip.has("image_grid_thw"));
  ASSERT_TRUE(roundtrip.has("aux"));
  expect_tensor_equal(
      std::get<torch::Tensor>(roundtrip.data().at("pixel_values")),
      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}}, torch::kFloat32));
  expect_tensor_equal(
      std::get<torch::Tensor>(roundtrip.data().at("image_grid_thw")),
      torch::tensor({1, 2, 3}, torch::kInt32));
  const auto& rt_aux =
      std::get<std::vector<torch::Tensor>>(roundtrip.data().at("aux"));
  ASSERT_EQ(rt_aux.size(), 2);
  expect_tensor_equal(rt_aux[0], torch::tensor({5, 6}, torch::kInt64));
  expect_tensor_equal(rt_aux[1], torch::tensor({7, 8}, torch::kInt32));

  const auto& rt_item = roundtrip.mm_data_vec()[0].items<MMItemVec>()[0];
  EXPECT_EQ(rt_item.type(), MMType::IMAGE);
  EXPECT_EQ(rt_item.state().token_pos().offset, 7u);
  EXPECT_EQ(rt_item.state().token_pos().length, 9u);
  EXPECT_EQ(rt_item.state().prefix_cache().cached_token_num, 3u);
  EXPECT_EQ(
      std::memcmp(rt_item.state().prefix_cache().key.data,
                  item.state().prefix_cache().key.data,
                  XXH3_128BITS_HASH_VALUE_LEN),
      0);

  ASSERT_TRUE(rt_item.has("pixel_values"));
  ASSERT_TRUE(rt_item.has("image_grid_thw"));
  ASSERT_TRUE(rt_item.has("aux"));
  expect_tensor_equal(std::get<torch::Tensor>(rt_item.data().at("pixel_values")),
                      torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}},
                                    torch::kFloat32));
  expect_tensor_equal(
      std::get<torch::Tensor>(rt_item.data().at("image_grid_thw")),
      torch::tensor({1, 2, 3}, torch::kInt32));
  const auto& aux =
      std::get<std::vector<torch::Tensor>>(rt_item.data().at("aux"));
  ASSERT_EQ(aux.size(), 2);
  expect_tensor_equal(aux[0], torch::tensor({5, 6}, torch::kInt64));
  expect_tensor_equal(aux[1], torch::tensor({7, 8}, torch::kInt32));
}

TEST(MMBatchDataProtoTest, KeepsLegacyDictOnlyRoundTrip) {
  MMDict dict;
  dict.emplace("pixel_values", torch::tensor({1.0f, 2.0f}, torch::kFloat32));
  dict.emplace("image_grid_thw", torch::tensor({1, 3}, torch::kInt32));

  MMBatchData batch(MMType::IMAGE, dict);
  proto::MMData pb_mm_data;
  ASSERT_TRUE(mmdata_to_proto(batch, &pb_mm_data));
  EXPECT_EQ(pb_mm_data.entries_size(), 0);

  MMBatchData roundtrip;
  ASSERT_TRUE(proto_to_mmdata(pb_mm_data, &roundtrip));
  EXPECT_TRUE(roundtrip.mm_data_vec().empty());
  ASSERT_TRUE(roundtrip.has("pixel_values"));
  ASSERT_TRUE(roundtrip.has("image_grid_thw"));
  expect_tensor_equal(
      std::get<torch::Tensor>(roundtrip.data().at("pixel_values")),
      torch::tensor({1.0f, 2.0f}, torch::kFloat32));
  expect_tensor_equal(
      std::get<torch::Tensor>(roundtrip.data().at("image_grid_thw")),
      torch::tensor({1, 3}, torch::kInt32));
}

}  // namespace
}  // namespace xllm
