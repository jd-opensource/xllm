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

MooncakeTransferEngine::BufferDesc mk_desc(int64_t group_id,
                                           int32_t slot_id,
                                           uint64_t addr,
                                           uint64_t len,
                                           int64_t bytes_per_block) {
  MooncakeTransferEngine::BufferDesc buffer_desc;
  buffer_desc.group_id = group_id;
  buffer_desc.slot_id = slot_id;
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
  ASSERT_EQ(mem_info.num_groups, 1);
  ASSERT_EQ(mem_info.groups.size(), 1U);
  EXPECT_EQ(mem_info.groups[0].slots.size(), 2U);
  EXPECT_EQ(mem_info.groups[0].slots.at(0).bytes_per_block, 0x10);
  EXPECT_EQ(mem_info.groups[0].slots.at(1).bytes_per_block, 0x10);
}

TEST(MooncakeTransferEngineTest, ParseDescInfo) {
  proto::RegisteredMemoryInfo pb_info;
  pb_info.set_num_layers(2);
  auto* slot0 = pb_info.add_buffers();
  slot0->set_group_id(0);
  slot0->set_slot_id(0);
  slot0->set_addr(0x1000);
  slot0->set_len(0x400);
  slot0->set_bytes_per_block(0x40);
  auto* slot2 = pb_info.add_buffers();
  slot2->set_group_id(0);
  slot2->set_slot_id(2);
  slot2->set_addr(0x2000);
  slot2->set_len(0x80);
  slot2->set_bytes_per_block(0x08);
  auto* slot1 = pb_info.add_buffers();
  slot1->set_group_id(1);
  slot1->set_slot_id(0);
  slot1->set_addr(0x3000);
  slot1->set_len(0x400);
  slot1->set_bytes_per_block(0x40);

  MooncakeTransferEngine::MemInfo mem_info;
  std::string err;
  ASSERT_TRUE(MooncakeTransferEngine::parse_mem_info(pb_info, &mem_info, &err))
      << err;
  ASSERT_EQ(mem_info.num_groups, 2);
  EXPECT_EQ(mem_info.groups[0].slots.size(), 2U);
  EXPECT_EQ(mem_info.groups[1].slots.size(), 1U);
  EXPECT_EQ(mem_info.groups[0].slots.at(2).addr, 0x2000U);
}

TEST(MooncakeTransferEngineTest, RejectBadSlotId) {
  proto::RegisteredMemoryInfo pb_info;
  pb_info.set_num_layers(1);
  auto* buf = pb_info.add_buffers();
  buf->set_group_id(0);
  buf->set_slot_id(-1);
  buf->set_addr(0x1000);
  buf->set_len(0x40);
  buf->set_bytes_per_block(0x40);

  MooncakeTransferEngine::MemInfo mem_info;
  std::string err;
  EXPECT_FALSE(
      MooncakeTransferEngine::parse_mem_info(pb_info, &mem_info, &err));
  EXPECT_NE(err.find("slot_id"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, RejectBadDescLen) {
  proto::RegisteredMemoryInfo pb_info;
  pb_info.set_num_layers(1);
  auto* buf = pb_info.add_buffers();
  buf->set_group_id(0);
  buf->set_slot_id(0);
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
  auto* slot0 = pb_info.add_buffers();
  slot0->set_group_id(0);
  slot0->set_slot_id(0);
  slot0->set_addr(0x1000);
  slot0->set_len(0x40);
  slot0->set_bytes_per_block(0x40);
  auto* dup0 = pb_info.add_buffers();
  dup0->set_group_id(0);
  dup0->set_slot_id(0);
  dup0->set_addr(0x2000);
  dup0->set_len(0x40);
  dup0->set_bytes_per_block(0x40);

  MooncakeTransferEngine::MemInfo mem_info;
  std::string err;
  EXPECT_FALSE(
      MooncakeTransferEngine::parse_mem_info(pb_info, &mem_info, &err));
  EXPECT_NE(err.find("duplicate"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, BuildEntriesForSlot0And2) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_groups = 1;
  remote_info.num_groups = 1;
  local_info.groups.resize(1);
  remote_info.groups.resize(1);

  local_info.groups[0].slots.emplace(0,
                                     mk_desc(/*group_id=*/0,
                                             /*slot_id=*/0,
                                             /*addr=*/0x1000,
                                             /*len=*/0x400,
                                             /*bytes_per_block=*/0x40));
  local_info.groups[0].slots.emplace(2,
                                     mk_desc(/*group_id=*/0,
                                             /*slot_id=*/2,
                                             /*addr=*/0x3000,
                                             /*len=*/0x80,
                                             /*bytes_per_block=*/0x08));
  remote_info.groups[0].slots.emplace(0,
                                      mk_desc(/*group_id=*/0,
                                              /*slot_id=*/0,
                                              /*addr=*/0x5000,
                                              /*len=*/0x400,
                                              /*bytes_per_block=*/0x40));
  remote_info.groups[0].slots.emplace(2,
                                      mk_desc(/*group_id=*/0,
                                              /*slot_id=*/2,
                                              /*addr=*/0x7000,
                                              /*len=*/0x80,
                                              /*bytes_per_block=*/0x08));
  local_info.buffers = {local_info.groups[0].slots.at(0),
                        local_info.groups[0].slots.at(2)};
  remote_info.buffers = {remote_info.groups[0].slots.at(0),
                         remote_info.groups[0].slots.at(2)};

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

TEST(MooncakeTransferEngineTest, RejectMismatchedSlotDesc) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_groups = 1;
  remote_info.num_groups = 1;
  local_info.groups.resize(1);
  remote_info.groups.resize(1);

  local_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x1000, 0x400, 0x40));
  local_info.groups[0].slots.emplace(2, mk_desc(0, 2, 0x3000, 0x80, 0x08));
  remote_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x5000, 0x400, 0x40));

  std::string err;
  EXPECT_FALSE(
      MooncakeTransferEngine::same_layout(local_info, remote_info, &err));
  EXPECT_NE(err.find("slot count mismatch"), std::string::npos);
}

TEST(MooncakeTransferEngineTest, AllowCapMismatchWithSameLayout) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_groups = 1;
  remote_info.num_groups = 1;
  local_info.groups.resize(1);
  remote_info.groups.resize(1);

  local_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x1000, 0x400, 0x40));
  remote_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x5000, 0x3c0, 0x40));

  std::string err;
  EXPECT_TRUE(
      MooncakeTransferEngine::same_layout(local_info, remote_info, &err))
      << err;
}

TEST(MooncakeTransferEngineTest, RejectEntryBounds) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_groups = 1;
  remote_info.num_groups = 1;
  local_info.groups.resize(1);
  remote_info.groups.resize(1);

  local_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x1000, 0x80, 0x40));
  remote_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x5000, 0x80, 0x40));
  local_info.buffers = {local_info.groups[0].slots.at(0)};
  remote_info.buffers = {remote_info.groups[0].slots.at(0)};

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
  local_info.num_groups = 1;
  remote_info.num_groups = 1;
  local_info.groups.resize(1);
  remote_info.groups.resize(1);

  local_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x1000, 0x400, 0x40));
  remote_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x5000, 0x100, 0x40));
  local_info.buffers = {local_info.groups[0].slots.at(0)};
  remote_info.buffers = {remote_info.groups[0].slots.at(0)};

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
  local_info.num_groups = 1;
  remote_info.num_groups = 1;
  local_info.groups.resize(1);
  remote_info.groups.resize(1);

  local_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x1000, 0x400, 0x40));
  remote_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x5000, 0x100, 0x40));
  local_info.buffers = {local_info.groups[0].slots.at(0)};
  remote_info.buffers = {remote_info.groups[0].slots.at(0)};

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

TEST(MooncakeTransferEngineTest, BuildEntriesForMultiGroup) {
  MooncakeTransferEngine::MemInfo local_info;
  MooncakeTransferEngine::MemInfo remote_info;
  local_info.num_groups = 2;
  remote_info.num_groups = 2;
  local_info.groups.resize(2);
  remote_info.groups.resize(2);

  local_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x1000, 0x400, 0x40));
  local_info.groups[1].slots.emplace(0, mk_desc(1, 0, 0x2000, 0x400, 0x40));
  local_info.groups[1].slots.emplace(2, mk_desc(1, 2, 0x3000, 0x80, 0x08));
  remote_info.groups[0].slots.emplace(0, mk_desc(0, 0, 0x5000, 0x400, 0x40));
  remote_info.groups[1].slots.emplace(0, mk_desc(1, 0, 0x6000, 0x400, 0x40));
  remote_info.groups[1].slots.emplace(2, mk_desc(1, 2, 0x7000, 0x80, 0x08));
  local_info.buffers = {local_info.groups[0].slots.at(0),
                        local_info.groups[1].slots.at(0),
                        local_info.groups[1].slots.at(2)};
  remote_info.buffers = {remote_info.groups[0].slots.at(0),
                         remote_info.groups[1].slots.at(0),
                         remote_info.groups[1].slots.at(2)};

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
