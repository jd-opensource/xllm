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

#include "composite_block_manager.h"

#include <gtest/gtest.h>

#include <set>

#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "framework/request/stopping_checker.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"

namespace xllm {

namespace {

constexpr uint32_t kManagerTypeBlockManagerImpl = 0;
constexpr uint32_t kManagerTypeSlidingWindowBlockManager = 1;

// Base block_size = 128. Two BlockManagerImpl: compress_ratio 4 and 128.
// - Ratio 4: block_size = 128*4 = 512, num_blocks = base_num_blocks/4.
// - Ratio 128: block_size = 128*128 = 16384, num_blocks = base_num_blocks/128.
// Use base_num_blocks = 128*32 = 4096 so that ratio-4 has 1024 blocks,
// ratio-128 has 32 blocks (ratio-4 block count is 32x ratio-128).
BlockManager::Options MakeCompositeOptions(uint32_t base_num_blocks,
                                           uint32_t block_size,
                                           uint32_t window_size,
                                           uint32_t max_seqs_per_batch) {
  BlockManager::Options opts;
  opts.num_blocks(base_num_blocks)
      .block_size(block_size)
      .window_size(window_size)
      .max_seqs_per_batch(max_seqs_per_batch)
      .manager_types({kManagerTypeSlidingWindowBlockManager,
                      kManagerTypeBlockManagerImpl,
                      kManagerTypeBlockManagerImpl})
      .compress_ratios({0, 4, 128});
  return opts;
}

constexpr uint32_t kBaseBlockSize = 128;
constexpr uint32_t kCompressRatio4 = 4;
constexpr uint32_t kCompressRatio128 = 128;
constexpr uint32_t kSlidingWindowBlocksPerSequence = 12;
// Sub-manager 1 (ratio 4): block_size = 128*4 = 512.
constexpr uint32_t kBlockSizeRatio4 = kBaseBlockSize * kCompressRatio4;
// Sub-manager 2 (ratio 128): block_size = 128*128 = 16384.
constexpr uint32_t kBlockSizeRatio128 = kBaseBlockSize * kCompressRatio128;

inline size_t CeilBlocks(size_t num_tokens, size_t block_size) {
  return (num_tokens + block_size - 1) / block_size;
}

// Creates a minimal Sequence for testing (same pattern as batch_test.cpp).
Sequence MakeTestSequence(size_t index,
                          const std::vector<int32_t>& prompt_token_ids) {
  torch::Device device(Device::type_torch(), 0);
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(256);
  SequenceParams seq_params;
  seq_params.seq_capacity = 5000;
  seq_params.stopping_checker = &stopping_checker;
  seq_params.sampling_param = &sampling_param;
  seq_params.skip_special_tokens = true;
  seq_params.echo = false;
  seq_params.logprobs = false;
  seq_params.enable_schedule_overlap = false;

  torch::Tensor input_embedding;
  MMData mm_data;
  IncrementalDecoder decoder("", 1, false, false);
  return Sequence(index,
                  prompt_token_ids,
                  input_embedding,
                  mm_data,
                  std::move(decoder),
                  seq_params);
}

}  // namespace

TEST(CompositeBlockManagerTest, AllocateForSequence_SingleSeq) {
  const uint32_t base_block_size = kBaseBlockSize;
  const uint32_t base_num_blocks = 4096;  // ratio-4: 1024 blocks, ratio-128: 32
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, base_block_size, window_size, max_seqs_per_batch);

  CompositeBlockManager manager(opts);
  EXPECT_TRUE(manager.is_composite());
  EXPECT_EQ(manager.num_sub_managers(), 3u);

  Sequence seq = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  const size_t num_tokens = 1024;

  EXPECT_TRUE(manager.allocate_for_sequence(&seq, num_tokens));

  const std::vector<std::vector<Block>>& composite =
      seq.kv_state().composite_blocks();
  ASSERT_EQ(composite.size(), 3u);

  // Sub-manager 0: SlidingWindow, fixed 12 blocks per sequence, block_size=128.
  EXPECT_EQ(composite[0].size(), kSlidingWindowBlocksPerSequence);
  for (const auto& b : composite[0]) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), base_block_size);
  }

  // Sub-manager 1: BlockManagerImpl compress_ratio 4, block_size=128*4=512.
  const size_t expected_blocks_1 = CeilBlocks(num_tokens, kBlockSizeRatio4);
  EXPECT_EQ(composite[1].size(), expected_blocks_1);
  for (const auto& b : composite[1]) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), kBlockSizeRatio4);
  }

  // Sub-manager 2: BlockManagerImpl compress_ratio 128,
  // block_size=128*128=16384.
  const size_t expected_blocks_2 = CeilBlocks(num_tokens, kBlockSizeRatio128);
  EXPECT_EQ(composite[2].size(), expected_blocks_2);
  for (const auto& b : composite[2]) {
    EXPECT_TRUE(b.is_valid());
    EXPECT_EQ(b.size(), kBlockSizeRatio128);
  }

  manager.deallocate_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_DifferentBatchSeqs) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(opts);

  // Seq1: 1024 tokens. Ratio 4: ceil(1024/512)=2; ratio 128:
  // ceil(1024/16384)=1.
  Sequence seq1 = MakeTestSequence(0, std::vector<int32_t>(1024, 1));
  EXPECT_TRUE(manager.allocate_for_sequence(&seq1, 1024));
  const auto& c1 = seq1.kv_state().composite_blocks();
  ASSERT_EQ(c1.size(), 3u);
  EXPECT_EQ(c1[0].size(), kSlidingWindowBlocksPerSequence);
  EXPECT_EQ(c1[1].size(), CeilBlocks(1024, kBlockSizeRatio4));
  EXPECT_EQ(c1[2].size(), CeilBlocks(1024, kBlockSizeRatio128));

  // Seq2: 1500 tokens. Ratio 4: ceil(1500/512)=3; ratio 128:
  // ceil(1500/16384)=1.
  Sequence seq2 = MakeTestSequence(1, std::vector<int32_t>(1500, 1));
  EXPECT_TRUE(manager.allocate_for_sequence(&seq2, 1500));
  const auto& c2 = seq2.kv_state().composite_blocks();
  ASSERT_EQ(c2.size(), 3u);
  EXPECT_EQ(c2[0].size(), kSlidingWindowBlocksPerSequence);
  EXPECT_EQ(c2[1].size(), CeilBlocks(1500, kBlockSizeRatio4));
  EXPECT_EQ(c2[2].size(), CeilBlocks(1500, kBlockSizeRatio128));

  // Blocks allocated to different seqs must not overlap (distinct block ids).
  std::set<int32_t> ids1_0, ids1_1, ids1_2, ids2_0, ids2_1, ids2_2;
  for (const auto& b : c1[0]) ids1_0.insert(b.id());
  for (const auto& b : c1[1]) ids1_1.insert(b.id());
  for (const auto& b : c1[2]) ids1_2.insert(b.id());
  for (const auto& b : c2[0]) ids2_0.insert(b.id());
  for (const auto& b : c2[1]) ids2_1.insert(b.id());
  for (const auto& b : c2[2]) ids2_2.insert(b.id());
  for (int32_t id : ids1_0) EXPECT_EQ(ids2_0.count(id), 0u);
  for (int32_t id : ids1_1) EXPECT_EQ(ids2_1.count(id), 0u);
  for (int32_t id : ids1_2) EXPECT_EQ(ids2_2.count(id), 0u);
  const int32_t max_swa_block_id =
      kSlidingWindowBlocksPerSequence * max_seqs_per_batch + 1;
  for (int32_t id : ids1_0) EXPECT_LE(id, max_swa_block_id);
  for (int32_t id : ids2_0) EXPECT_LE(id, max_swa_block_id);

  manager.deallocate_sequence(&seq1);
  manager.deallocate_sequence(&seq2);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_GrowSameSeq) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(opts);

  Sequence seq = MakeTestSequence(0, {1, 2, 3});
  // 600 tokens: ratio 4 needs ceil(600/512)=2 blocks, ratio 128 needs 1 block.
  EXPECT_TRUE(manager.allocate_for_sequence(&seq, 600));
  const std::vector<std::vector<Block>>& c0 = seq.kv_state().composite_blocks();
  ASSERT_EQ(c0.size(), 3u);
  EXPECT_EQ(c0[0].size(), kSlidingWindowBlocksPerSequence);
  EXPECT_EQ(c0[1].size(), CeilBlocks(600, kBlockSizeRatio4));
  EXPECT_EQ(c0[2].size(), CeilBlocks(600, kBlockSizeRatio128));

  // Grow to 1200 tokens: ratio 4 needs 3 blocks, ratio 128 still 1 block.
  EXPECT_TRUE(manager.allocate_for_sequence(&seq, 1200));
  const std::vector<std::vector<Block>>& c1 = seq.kv_state().composite_blocks();
  EXPECT_EQ(c1[0].size(), kSlidingWindowBlocksPerSequence);
  EXPECT_EQ(c1[1].size(), CeilBlocks(1200, kBlockSizeRatio4));
  EXPECT_EQ(c1[2].size(), CeilBlocks(1200, kBlockSizeRatio128));

  // No growth: still 1200 tokens, block counts unchanged.
  EXPECT_TRUE(manager.allocate_for_sequence(&seq, 1200));
  const std::vector<std::vector<Block>>& c2 = seq.kv_state().composite_blocks();
  EXPECT_EQ(c2[1].size(), CeilBlocks(1200, kBlockSizeRatio4));
  EXPECT_EQ(c2[2].size(), CeilBlocks(1200, kBlockSizeRatio128));

  manager.deallocate_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateContinuesAfterSatisfiedTokenManager) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  opts.compress_ratios({0, 128, 4});
  CompositeBlockManager manager(opts);

  Sequence seq = MakeTestSequence(0, {1});
  EXPECT_TRUE(manager.allocate_for_sequence(&seq, 1024));
  const auto& c0 = seq.kv_state().composite_blocks();
  ASSERT_EQ(c0.size(), 3u);
  EXPECT_EQ(c0[1].size(), CeilBlocks(1024, kBlockSizeRatio128));
  EXPECT_EQ(c0[2].size(), CeilBlocks(1024, kBlockSizeRatio4));

  EXPECT_TRUE(manager.allocate_for_sequence(&seq, 1500));
  const auto& c1 = seq.kv_state().composite_blocks();
  EXPECT_EQ(c1[1].size(), CeilBlocks(1500, kBlockSizeRatio128));
  EXPECT_EQ(c1[2].size(), CeilBlocks(1500, kBlockSizeRatio4));

  manager.deallocate_sequence(&seq);
}

TEST(CompositeBlockManagerTest, AllocateForSequence_NullSeqReturnsFalse) {
  BlockManager::Options opts =
      MakeCompositeOptions(4096, kBaseBlockSize, 128, 4);
  CompositeBlockManager manager(opts);
  EXPECT_FALSE(manager.allocate_for_sequence(nullptr, 10));
}

// Verifies that when seq token count increases, CompositeBlockManager correctly
// adds blocks (only appends new blocks; existing block ids are preserved).
TEST(CompositeBlockManagerTest, TokenIncrease_AddsBlocksIncrementally) {
  const uint32_t base_num_blocks = 4096;
  const uint32_t window_size = 128;
  const uint32_t max_seqs_per_batch = 4;

  BlockManager::Options opts = MakeCompositeOptions(
      base_num_blocks, kBaseBlockSize, window_size, max_seqs_per_batch);
  CompositeBlockManager manager(opts);

  Sequence seq = MakeTestSequence(0, {1});
  std::vector<size_t> token_steps = {100, 600, 1200, 2000, 2500};

  std::vector<std::set<int32_t>> prev_ids_1;
  std::vector<std::set<int32_t>> prev_ids_2;

  for (size_t num_tokens : token_steps) {
    EXPECT_TRUE(manager.allocate_for_sequence(&seq, num_tokens));

    const std::vector<std::vector<Block>>& composite =
        seq.kv_state().composite_blocks();
    ASSERT_EQ(composite.size(), 3u);

    const size_t expect_1 = CeilBlocks(num_tokens, kBlockSizeRatio4);
    const size_t expect_2 = CeilBlocks(num_tokens, kBlockSizeRatio128);
    EXPECT_EQ(composite[1].size(), expect_1)
        << "num_tokens=" << num_tokens << " layer1 block count";
    EXPECT_EQ(composite[2].size(), expect_2)
        << "num_tokens=" << num_tokens << " layer2 block count";

    // Check that previously allocated block ids are still present (only
    // append).
    std::set<int32_t> ids_1, ids_2;
    for (const auto& b : composite[1]) ids_1.insert(b.id());
    for (const auto& b : composite[2]) ids_2.insert(b.id());
    for (const auto& prev : prev_ids_1) {
      for (int32_t id : prev) EXPECT_GT(ids_1.count(id), 0u);
    }
    for (const auto& prev : prev_ids_2) {
      for (int32_t id : prev) EXPECT_GT(ids_2.count(id), 0u);
    }
    prev_ids_1.push_back(ids_1);
    prev_ids_2.push_back(ids_2);
  }

  manager.deallocate_sequence(&seq);
}

}  // namespace xllm
