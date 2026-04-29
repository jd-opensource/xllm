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

#include "framework/batch/transfer_kv_info_builder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace xllm {

namespace {

TransferKVInfo make_info(const std::vector<uint64_t>& remote_block_ids) {
  TransferKVInfo info;
  info.request_id = "req_0";
  info.remote_blocks_ids = remote_block_ids;
  info.dp_rank = 3;
  return info;
}

XTensorLayerOffsets make_offsets(const std::vector<uint64_t>& k_offsets,
                                 const std::vector<uint64_t>& v_offsets) {
  XTensorLayerOffsets offsets;
  offsets.k_offsets = k_offsets;
  offsets.v_offsets = v_offsets;
  return offsets;
}

void expect_blocks(const TransferKVInfo& info,
                   const std::vector<uint64_t>& local_block_ids,
                   const std::vector<uint64_t>& remote_block_ids) {
  EXPECT_EQ(info.local_blocks_ids, local_block_ids);
  EXPECT_EQ(info.remote_blocks_ids, remote_block_ids);
  EXPECT_EQ(info.request_id, "req_0");
  EXPECT_EQ(info.dp_rank, 3);
}

}  // namespace

TEST(TransferKVInfoBuilderTest, FirstChunkUsesRemotePrefix) {
  const TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});

  const TransferKVInfo info =
      build_step_transfer_info(full_info,
                               /*local_block_ids=*/{10, 11},
                               /*n_kv_cache_tokens=*/0,
                               /*seq_len=*/32,
                               /*block_size=*/16);

  expect_blocks(info, {10, 11}, {100, 101});
}

TEST(TransferKVInfoBuilderTest, LaterChunkUsesLogicalOffset) {
  const TransferKVInfo full_info = make_info({100, 101, 102, 103, 104});

  const TransferKVInfo info =
      build_step_transfer_info(full_info,
                               /*local_block_ids=*/{10, 11, 12, 13},
                               /*n_kv_cache_tokens=*/32,
                               /*seq_len=*/64,
                               /*block_size=*/16);

  expect_blocks(info, {12, 13}, {102, 103});
}

TEST(TransferKVInfoBuilderTest, PartialBoundaryRepeatsDirtyBlocks) {
  const TransferKVInfo full_info = make_info({100, 101, 102});

  const TransferKVInfo info =
      build_step_transfer_info(full_info,
                               /*local_block_ids=*/{10, 11, 12},
                               /*n_kv_cache_tokens=*/15,
                               /*seq_len=*/33,
                               /*block_size=*/16);

  expect_blocks(info, {10, 11, 12}, {100, 101, 102});
}

TEST(TransferKVInfoBuilderTest, SharedPrefixKeepsExistingMapping) {
  const TransferKVInfo full_info = make_info({102, 103, 104});

  const TransferKVInfo info =
      build_step_transfer_info(full_info,
                               /*local_block_ids=*/{10, 11, 12, 13, 14},
                               /*n_kv_cache_tokens=*/0,
                               /*seq_len=*/80,
                               /*block_size=*/16);

  expect_blocks(info, {12, 13, 14}, {102, 103, 104});
}

TEST(TransferKVInfoBuilderTest, SharedPrefixSlicesXTensorOffsets) {
  TransferKVInfo full_info = make_info({102, 103, 104});
  full_info.local_blocks_ids = {0, 0, 0, 0, 0};
  full_info.dst_xtensor_layer_offsets = {
      make_offsets({1000, 1001, 1002}, {2000, 2001, 2002}),
      make_offsets({3000, 3001, 3002}, {4000, 4001, 4002})};

  const TransferKVInfo info =
      build_step_transfer_info(full_info,
                               /*local_block_ids=*/{10, 11, 12, 13},
                               /*n_kv_cache_tokens=*/32,
                               /*seq_len=*/64,
                               /*block_size=*/16);

  expect_blocks(info, {12, 13}, {102, 103});
  ASSERT_EQ(info.dst_xtensor_layer_offsets.size(), 2u);
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].k_offsets,
            (std::vector<uint64_t>{1000, 1001}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].v_offsets,
            (std::vector<uint64_t>{2000, 2001}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[1].k_offsets,
            (std::vector<uint64_t>{3000, 3001}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[1].v_offsets,
            (std::vector<uint64_t>{4000, 4001}));
}

TEST(TransferKVInfoBuilderTest, PartialBoundaryRepeatsXTensorOffsets) {
  TransferKVInfo full_info = make_info({100, 101, 102});
  full_info.dst_xtensor_layer_offsets = {
      make_offsets({1000, 1001, 1002}, {2000, 2001, 2002}),
      make_offsets({3000, 3001, 3002}, {4000, 4001, 4002})};

  const TransferKVInfo info =
      build_step_transfer_info(full_info,
                               /*local_block_ids=*/{10, 11, 12},
                               /*n_kv_cache_tokens=*/15,
                               /*seq_len=*/33,
                               /*block_size=*/16);

  expect_blocks(info, {10, 11, 12}, {100, 101, 102});
  ASSERT_EQ(info.dst_xtensor_layer_offsets.size(), 2u);
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].k_offsets,
            (std::vector<uint64_t>{1000, 1001, 1002}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[0].v_offsets,
            (std::vector<uint64_t>{2000, 2001, 2002}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[1].k_offsets,
            (std::vector<uint64_t>{3000, 3001, 3002}));
  EXPECT_EQ(info.dst_xtensor_layer_offsets[1].v_offsets,
            (std::vector<uint64_t>{4000, 4001, 4002}));
}

}  // namespace xllm
