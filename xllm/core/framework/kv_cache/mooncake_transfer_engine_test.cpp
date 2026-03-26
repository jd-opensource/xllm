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

#include "mooncake_transfer_engine.h"

#include <gtest/gtest.h>

namespace xllm {

namespace {

MooncakeTransferEngine::BufferDesc mk_desc(int64_t layer_id,
                                           proto::BufferKind kind,
                                           uint64_t addr,
                                           uint64_t len,
                                           int64_t bytes_per_block) {
  MooncakeTransferEngine::BufferDesc buffer_desc;
  buffer_desc.layer_id = layer_id;
  buffer_desc.kind = kind;
  buffer_desc.addr = addr;
  buffer_desc.len = len;
  buffer_desc.bytes_per_block = bytes_per_block;
  return buffer_desc;
}

}  // namespace

TEST(MooncakeTransferEngineTest, ParseLegacyPairInfo) {
  proto::RegisteredMemoryInfo pb_info;
  pb_info.add_addrs(0x1000);
  pb_info.add_addrs(0x2000);
  pb_info.add_lens(0x100);
  pb_info.add_lens(0x100);
  pb_info.set_size_per_block(0x10);
  pb_info.set_num_layers(1);

  MooncakeTransferEngine::MemInfo mem_info;
  std::string err;
  ASSERT_TRUE(MooncakeTransferEngine::parse_mem_info(pb_info, &mem_info, &err))
      << err;
  ASSERT_EQ(mem_info.num_layers, 1);
  ASSERT_EQ(mem_info.layers.size(), 1U);
  ASSERT_TRUE(mem_info.layers[0].key.has_value());
  ASSERT_TRUE(mem_info.layers[0].value.has_value());
  EXPECT_FALSE(mem_info.layers[0].index.has_value());
  EXPECT_EQ(mem_info.layers[0].key->bytes_per_block, 0x10);
  EXPECT_EQ(mem_info.layers[0].value->bytes_per_block, 0x10);
}

TEST(MooncakeTransferEngineTest, ParseDescInfo) {
  proto::RegisteredMemoryInfo pb_info;
  pb_info.set_num_layers(2);
  auto* key0 = pb_info.add_buffers();
  key0->set_layer_id(0);
  key0->set_kind(proto::BUFFER_KIND_KEY);
  key0->set_addr(0x1000);
  key0->set_len(0x400);
  key0->set_bytes_per_block(0x40);
  auto* idx0 = pb_info.add_buffers();
  idx0->set_layer_id(0);
  idx0->set_kind(proto::BUFFER_KIND_INDEX);
  idx0->set_addr(0x2000);
  idx0->set_len(0x80);
  idx0->set_bytes_per_block(0x08);
  auto* key1 = pb_info.add_buffers();
  key1->set_layer_id(1);
  key1->set_kind(proto::BUFFER_KIND_KEY);
  key1->set_addr(0x3000);
  key1->set_len(0x400);
  key1->set_bytes_per_block(0x40);

  MooncakeTransferEngine::MemInfo mem_info;
  std::string err;
  ASSERT_TRUE(MooncakeTransferEngine::parse_mem_info(pb_info, &mem_info, &err))
      << err;
  ASSERT_EQ(mem_info.num_layers, 2);
  ASSERT_TRUE(mem_info.layers[0].key.has_value());
  ASSERT_TRUE(mem_info.layers[0].index.has_value());
  ASSERT_TRUE(mem_info.layers[1].key.has_value());
  EXPECT_FALSE(mem_info.layers[1].value.has_value());
}

TEST(MooncakeTransferEngineTest, RejectUnknownKind) {
  proto::RegisteredMemoryInfo pb_info;
  pb_info.set_num_layers(1);
  auto* buf = pb_info.add_buffers();
  buf->set_layer_id(0);
  buf->set_kind(proto::BUFFER_KIND_UNKNOWN);
  buf->set_addr(0x1000);
  buf->set_len(0x40);
  buf->set_bytes_per_block(0x40);

  MooncakeTransferEngine::MemInfo mem_info;
  std::string err;
  EXPECT_FALSE(
      MooncakeTransferEngine::parse_mem_info(pb_info, &mem_info, &err));
  EXPECT_NE(err.find("buffer kind unknown"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, RejectBadDescLen) {
  proto::RegisteredMemoryInfo pb_info;
  pb_info.set_num_layers(1);
  auto* buf = pb_info.add_buffers();
  buf->set_layer_id(0);
  buf->set_kind(proto::BUFFER_KIND_KEY);
  buf->set_addr(0x1000);
  buf->set_len(0x41);
  buf->set_bytes_per_block(0x40);

  MooncakeTransferEngine::MemInfo mem_info;
  std::string err;
  EXPECT_FALSE(
      MooncakeTransferEngine::parse_mem_info(pb_info, &mem_info, &err));
  EXPECT_NE(err.find("align"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, RejectDupDesc) {
  proto::RegisteredMemoryInfo pb_info;
  pb_info.set_num_layers(1);
  auto* key0 = pb_info.add_buffers();
  key0->set_layer_id(0);
  key0->set_kind(proto::BUFFER_KIND_KEY);
  key0->set_addr(0x1000);
  key0->set_len(0x40);
  key0->set_bytes_per_block(0x40);
  auto* key1 = pb_info.add_buffers();
  key1->set_layer_id(0);
  key1->set_kind(proto::BUFFER_KIND_KEY);
  key1->set_addr(0x2000);
  key1->set_len(0x40);
  key1->set_bytes_per_block(0x40);

  MooncakeTransferEngine::MemInfo mem_info;
  std::string err;
  EXPECT_FALSE(
      MooncakeTransferEngine::parse_mem_info(pb_info, &mem_info, &err));
  EXPECT_NE(err.find("duplicate"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, BuildEntriesForKeyAndIndex) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_layers = 1;
  remote_info.num_layers = 1;
  local_info.layers.resize(1);
  remote_info.layers.resize(1);

  local_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x1000, 0x400, 0x40);
  local_info.layers[0].index =
      mk_desc(0, proto::BUFFER_KIND_INDEX, 0x3000, 0x80, 0x08);
  remote_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x5000, 0x400, 0x40);
  remote_info.layers[0].index =
      mk_desc(0, proto::BUFFER_KIND_INDEX, 0x7000, 0x80, 0x08);

  std::vector<TransferRequest> entries;
  std::string err;
  ASSERT_TRUE(MooncakeTransferEngine::build_entries(
      local_info,
      remote_info,
      {1},
      {3},
      {2},
      {0},
      MooncakeTransferEngine::MoveOpcode::READ,
      /*remote_handle=*/9,
      &entries,
      &err))
      << err;
  ASSERT_EQ(entries.size(), 2U);
  EXPECT_EQ(entries[0].length, 0x80U);
  EXPECT_EQ(entries[1].length, 0x10U);
  EXPECT_EQ(reinterpret_cast<uint64_t>(entries[0].source), 0x10c0U);
  EXPECT_EQ(entries[0].target_offset, 0x5040U);
  EXPECT_EQ(reinterpret_cast<uint64_t>(entries[1].source), 0x3018U);
  EXPECT_EQ(entries[1].target_offset, 0x7008U);
}

TEST(MooncakeTransferEngineTest, RejectMismatchedIndexDesc) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_layers = 1;
  remote_info.num_layers = 1;
  local_info.layers.resize(1);
  remote_info.layers.resize(1);

  local_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x1000, 0x400, 0x40);
  local_info.layers[0].index =
      mk_desc(0, proto::BUFFER_KIND_INDEX, 0x3000, 0x80, 0x08);
  remote_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x5000, 0x400, 0x40);

  std::string err;
  EXPECT_FALSE(
      MooncakeTransferEngine::same_layout(local_info, remote_info, &err));
  EXPECT_NE(err.find("kind mismatch"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, AllowCapMismatchWithSameLayout) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_layers = 1;
  remote_info.num_layers = 1;
  local_info.layers.resize(1);
  remote_info.layers.resize(1);

  local_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x1000, 0x400, 0x40);
  remote_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x5000, 0x3c0, 0x40);

  std::string err;
  EXPECT_TRUE(
      MooncakeTransferEngine::same_layout(local_info, remote_info, &err))
      << err;
}

TEST(MooncakeTransferEngineTest, RejectEntryBounds) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_layers = 1;
  remote_info.num_layers = 1;
  local_info.layers.resize(1);
  remote_info.layers.resize(1);

  local_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x1000, 0x80, 0x40);
  remote_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x5000, 0x80, 0x40);

  std::vector<TransferRequest> entries;
  std::string err;
  EXPECT_FALSE(MooncakeTransferEngine::build_entries(
      local_info,
      remote_info,
      {1},
      {1},
      {2},
      {0},
      MooncakeTransferEngine::MoveOpcode::READ,
      /*remote_handle=*/9,
      &entries,
      &err));
  EXPECT_NE(err.find("local range overflow"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, BuildEntriesWithAsymCap) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_layers = 1;
  remote_info.num_layers = 1;
  local_info.layers.resize(1);
  remote_info.layers.resize(1);

  local_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x1000, 0x400, 0x40);
  remote_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x5000, 0x100, 0x40);

  std::vector<TransferRequest> entries;
  std::string err;
  ASSERT_TRUE(MooncakeTransferEngine::build_entries(
      local_info,
      remote_info,
      {1, 2},
      {10, 11},
      {1, 1},
      {0},
      MooncakeTransferEngine::MoveOpcode::READ,
      /*remote_handle=*/9,
      &entries,
      &err))
      << err;
  ASSERT_EQ(entries.size(), 2U);
  EXPECT_EQ(reinterpret_cast<uint64_t>(entries[0].source), 0x1280U);
  EXPECT_EQ(entries[0].target_offset, 0x5040U);
  EXPECT_EQ(reinterpret_cast<uint64_t>(entries[1].source), 0x12c0U);
  EXPECT_EQ(entries[1].target_offset, 0x5080U);
}

TEST(MooncakeTransferEngineTest, RejectRemoteRangeOverflow) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_layers = 1;
  remote_info.num_layers = 1;
  local_info.layers.resize(1);
  remote_info.layers.resize(1);

  local_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x1000, 0x400, 0x40);
  remote_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x5000, 0x100, 0x40);

  std::vector<TransferRequest> entries;
  std::string err;
  EXPECT_FALSE(MooncakeTransferEngine::build_entries(
      local_info,
      remote_info,
      {0},
      {4},
      {1},
      {0},
      MooncakeTransferEngine::MoveOpcode::WRITE,
      /*remote_handle=*/9,
      &entries,
      &err));
  EXPECT_NE(err.find("remote range overflow"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, BuildEntriesForMultiLayer) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_layers = 2;
  remote_info.num_layers = 2;
  local_info.layers.resize(2);
  remote_info.layers.resize(2);

  local_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x1000, 0x400, 0x40);
  local_info.layers[1].key =
      mk_desc(1, proto::BUFFER_KIND_KEY, 0x2000, 0x400, 0x40);
  local_info.layers[1].index =
      mk_desc(1, proto::BUFFER_KIND_INDEX, 0x3000, 0x80, 0x08);
  remote_info.layers[0].key =
      mk_desc(0, proto::BUFFER_KIND_KEY, 0x5000, 0x400, 0x40);
  remote_info.layers[1].key =
      mk_desc(1, proto::BUFFER_KIND_KEY, 0x6000, 0x400, 0x40);
  remote_info.layers[1].index =
      mk_desc(1, proto::BUFFER_KIND_INDEX, 0x7000, 0x80, 0x08);

  std::vector<TransferRequest> entries;
  std::string err;
  ASSERT_TRUE(MooncakeTransferEngine::build_entries(
      local_info,
      remote_info,
      {0, 2},
      {4, 6},
      {1, 2},
      {},
      MooncakeTransferEngine::MoveOpcode::READ,
      /*remote_handle=*/9,
      &entries,
      &err))
      << err;
  ASSERT_EQ(entries.size(), 6U);
  EXPECT_EQ(entries[0].length, 0x40U);
  EXPECT_EQ(entries[1].length, 0x80U);
  EXPECT_EQ(entries[2].length, 0x40U);
  EXPECT_EQ(entries[3].length, 0x80U);
  EXPECT_EQ(entries[4].length, 0x08U);
  EXPECT_EQ(entries[5].length, 0x10U);
}

}  // namespace xllm
