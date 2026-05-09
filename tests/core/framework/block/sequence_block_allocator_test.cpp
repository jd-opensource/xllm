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

#include "framework/block/sequence_block_allocator.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "framework/block/block_group.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence.h"

namespace xllm {
namespace {

BlockGroupSpec make_spec(int32_t group_id,
                         BlockGroupKind kind,
                         int32_t tokens_per_block,
                         int64_t num_blocks,
                         int32_t fixed_blocks_per_sequence) {
  BlockGroupSpec spec;
  spec.group_id = group_id;
  spec.kind = kind;
  spec.tokens_per_block = tokens_per_block;
  spec.num_blocks = num_blocks;
  spec.fixed_blocks_per_sequence = fixed_blocks_per_sequence;
  return spec;
}

CompositeBlockPlan make_plan() {
  CompositeBlockPlan plan;
  plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::RING,
                /*tokens_per_block=*/16,
                /*num_blocks=*/4,
                /*fixed_blocks_per_sequence=*/2),
      make_spec(/*group_id=*/1,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/4,
                /*num_blocks=*/8,
                /*fixed_blocks_per_sequence=*/0),
      make_spec(/*group_id=*/2,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/10,
                /*num_blocks=*/3,
                /*fixed_blocks_per_sequence=*/0),
  };
  return plan;
}

CompositeBlockPlan make_small_plan() {
  CompositeBlockPlan plan;
  plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::RING,
                /*tokens_per_block=*/16,
                /*num_blocks=*/2,
                /*fixed_blocks_per_sequence=*/2),
      make_spec(/*group_id=*/1,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/4,
                /*num_blocks=*/2,
                /*fixed_blocks_per_sequence=*/0),
      make_spec(/*group_id=*/2,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/10,
                /*num_blocks=*/1,
                /*fixed_blocks_per_sequence=*/0),
  };
  return plan;
}

std::vector<int32_t> make_tokens(int32_t first, int32_t count) {
  std::vector<int32_t> tokens;
  tokens.reserve(count);
  for (int32_t i = 0; i < count; ++i) {
    tokens.emplace_back(first + i);
  }
  return tokens;
}

std::vector<int32_t> make_tokens_with_suffix(int32_t prefix_first,
                                             int32_t prefix_count,
                                             int32_t suffix_first,
                                             int32_t suffix_count) {
  std::vector<int32_t> tokens;
  tokens.reserve(static_cast<size_t>(prefix_count + suffix_count));
  for (int32_t i = 0; i < prefix_count; ++i) {
    tokens.emplace_back(prefix_first + i);
  }
  for (int32_t i = 0; i < suffix_count; ++i) {
    tokens.emplace_back(suffix_first + i);
  }
  return tokens;
}

CompositeBlockPlan make_composite_sum_plan() {
  CompositeBlockPlan plan;
  plan.groups = {
      make_spec(/*group_id=*/0,
                BlockGroupKind::RING,
                /*tokens_per_block=*/16,
                /*num_blocks=*/4,
                /*fixed_blocks_per_sequence=*/1),
      make_spec(/*group_id=*/1,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/4,
                /*num_blocks=*/6,
                /*fixed_blocks_per_sequence=*/0),
      make_spec(/*group_id=*/2,
                BlockGroupKind::TOKEN,
                /*tokens_per_block=*/8,
                /*num_blocks=*/4,
                /*fixed_blocks_per_sequence=*/0),
  };
  return plan;
}

class CompositeSequenceBlockAllocatorTest : public testing::Test {
 protected:
  Sequence make_sequence(size_t index, const std::vector<int32_t>& tokens) {
    SequenceParams params;
    params.seq_capacity = tokens.size() + 32;
    params.echo = false;
    params.skip_special_tokens = true;
    params.streaming = false;
    params.enable_schedule_overlap = false;
    params.rec_type = RecType::kNone;
    params.bos_token_id = 0;
    params.request_id = "composite_allocator_test";
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;

    IncrementalDecoder decoder(
        /*prompt=*/"prompt",
        /*num_prompt_tokens=*/tokens.size(),
        /*echo=*/params.echo,
        /*skip_special_tokens=*/params.skip_special_tokens);

    return Sequence(index,
                    tokens,
                    /*input_embedding=*/torch::Tensor(),
                    /*mm_data=*/MMData(),
                    decoder,
                    params);
  }

 private:
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

TEST_F(CompositeSequenceBlockAllocatorTest,
       AllocatesRingOnceAndTokenGroupsByTargetTokens) {
  CompositeSequenceBlockAllocator allocator(make_plan());
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});

  ASSERT_TRUE(allocator.allocate_sequence(&sequence, /*target_num_tokens=*/9));

  KVCacheState& kv_state = sequence.kv_state();
  EXPECT_TRUE(kv_state.has_composite_blocks());
  ASSERT_EQ(kv_state.num_composite_groups(), 3);
  EXPECT_EQ(kv_state.num_composite_blocks(/*group_id=*/0), 2);
  EXPECT_EQ(kv_state.num_composite_blocks(/*group_id=*/1), 3);
  EXPECT_EQ(kv_state.num_composite_blocks(/*group_id=*/2), 1);
  EXPECT_EQ(kv_state.num_kv_blocks(), 0);
  EXPECT_EQ(kv_state.token_capacity(), 10);

  ASSERT_TRUE(allocator.allocate_sequence(&sequence, /*target_num_tokens=*/17));

  EXPECT_EQ(kv_state.num_composite_blocks(/*group_id=*/0), 2);
  EXPECT_EQ(kv_state.num_composite_blocks(/*group_id=*/1), 5);
  EXPECT_EQ(kv_state.num_composite_blocks(/*group_id=*/2), 2);
  EXPECT_EQ(kv_state.num_kv_blocks(), 0);
  EXPECT_EQ(kv_state.token_capacity(), 20);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       FailedAllocationLeavesSequenceStateUnchanged) {
  CompositeSequenceBlockAllocator allocator(make_small_plan());
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});

  EXPECT_FALSE(allocator.allocate_sequence(&sequence, /*target_num_tokens=*/9));

  KVCacheState& kv_state = sequence.kv_state();
  EXPECT_FALSE(kv_state.has_composite_blocks());
  EXPECT_EQ(kv_state.num_composite_groups(), 0);
  EXPECT_EQ(kv_state.total_composite_blocks(), 0);
  EXPECT_EQ(kv_state.num_kv_blocks(), 0);
  EXPECT_EQ(kv_state.token_capacity(), 0);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       DeallocateReleasesBlocksWithoutResettingSequence) {
  CompositeSequenceBlockAllocator allocator(make_plan());
  Sequence sequence = make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3});

  ASSERT_TRUE(allocator.allocate_sequence(&sequence, /*target_num_tokens=*/9));
  sequence.kv_state().set_kv_cache_tokens_num(/*num=*/7);

  allocator.deallocate_sequence(&sequence);

  KVCacheState& kv_state = sequence.kv_state();
  EXPECT_FALSE(kv_state.has_composite_blocks());
  EXPECT_EQ(kv_state.total_composite_blocks(), 0);
  EXPECT_EQ(kv_state.kv_cache_tokens_num(), 7);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       PoolFailureClearsNewlyAssignedDpRank) {
  BlockManagerPool::Options options;
  options.num_blocks(1).block_size(4).enable_prefix_cache(false);
  BlockManagerPool pool(options, /*dp_size=*/1);
  Sequence sequence =
      make_sequence(/*index=*/0, /*tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});

  EXPECT_FALSE(pool.allocate(&sequence));
  EXPECT_EQ(sequence.dp_rank(), -1);
  EXPECT_FALSE(sequence.has_single_block_id());
  EXPECT_EQ(sequence.kv_state().num_kv_blocks(), 0);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       SingleGroupEstimateReleaseSkipsContendedPrefixBlocks) {
  BlockManagerPool::Options options;
  options.num_blocks(5).block_size(4).enable_prefix_cache(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seed_sequence =
      make_sequence(/*index=*/0, make_tokens(/*first=*/1, /*count=*/9));
  const bool seed_allocated = pool.allocate(&seed_sequence);
  EXPECT_TRUE(seed_allocated);
  if (seed_allocated) {
    seed_sequence.kv_state().set_kv_cache_tokens_num(
        seed_sequence.num_tokens());
    pool.cache(&seed_sequence);
  }

  Sequence shared_sequence =
      make_sequence(/*index=*/1, make_tokens(/*first=*/1, /*count=*/9));
  const bool shared_allocated = pool.allocate(&shared_sequence);
  EXPECT_TRUE(shared_allocated);

  if (seed_allocated && shared_allocated) {
    const bool shared_shape_ok =
        shared_sequence.kv_state().num_kv_blocks() == 3 &&
        shared_sequence.kv_state().shared_kv_blocks_num() == 2;
    EXPECT_TRUE(shared_shape_ok);
    if (shared_shape_ok) {
      EXPECT_GT(shared_sequence.kv_state().kv_blocks()[0].ref_count(), 2u);
      EXPECT_GT(shared_sequence.kv_state().kv_blocks()[1].ref_count(), 2u);
      EXPECT_LE(shared_sequence.kv_state().kv_blocks()[2].ref_count(), 2u);
    }
    const std::vector<size_t> free_blocks = pool.num_free_blocks();
    EXPECT_EQ(free_blocks.size(), 1);
    if (!free_blocks.empty()) {
      EXPECT_EQ(free_blocks[0], 0);
    }

    const std::vector<BlockGroupUsage> release =
        pool.estimate_release(&shared_sequence);
    EXPECT_EQ(release.size(), 1);
    if (!release.empty()) {
      EXPECT_EQ(release[0].releasable_blocks, 1)
          << "Only the unique suffix block should be releasable; shared prefix "
             "blocks still held by seed_sequence and prefix cache must not be "
             "used for eviction capacity.";
    }

    Sequence target_sequence =
        make_sequence(/*index=*/2, make_tokens(/*first=*/100, /*count=*/8));
    std::vector<Sequence*> release_candidates = {&shared_sequence};
    EXPECT_FALSE(pool.can_allocate_after_release(
        &target_sequence, target_sequence.num_tokens(), release_candidates))
        << "Scheduler eviction planning must not accept a target that needs "
           "two blocks when the selected candidate can actually release only "
           "one non-contended block.";
  }

  if (shared_sequence.dp_rank() >= 0) {
    pool.deallocate_without_cache(&shared_sequence);
  }
  if (seed_sequence.dp_rank() >= 0) {
    pool.deallocate_without_cache(&seed_sequence);
  }

  int32_t dp_rank = -1;
  const size_t cleanup_tokens = static_cast<size_t>(options.num_blocks() - 1) *
                                static_cast<size_t>(options.block_size());
  std::vector<Block> cleanup_blocks = pool.allocate(cleanup_tokens, dp_rank);
  EXPECT_EQ(cleanup_blocks.size(), options.num_blocks() - 1);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       SingleGroupReleaseCandidatesDoNotDoubleCountSharedBlock) {
  BlockManagerPool::Options options;
  options.num_blocks(3).block_size(4).enable_prefix_cache(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence first_candidate =
      make_sequence(/*index=*/0, make_tokens(/*first=*/1, /*count=*/4));
  ASSERT_TRUE(pool.allocate(&first_candidate));
  ASSERT_EQ(first_candidate.dp_rank(), 0);
  ASSERT_EQ(first_candidate.kv_state().num_kv_blocks(), 1);

  Sequence second_candidate =
      make_sequence(/*index=*/1, make_tokens(/*first=*/10, /*count=*/4));
  second_candidate.set_dp_rank(first_candidate.dp_rank());
  {
    std::vector<Block> shared_blocks = first_candidate.kv_state().kv_blocks();
    second_candidate.add_kv_blocks(shared_blocks);
  }

  ASSERT_EQ(second_candidate.kv_state().num_kv_blocks(), 1);
  EXPECT_EQ(first_candidate.kv_state().kv_blocks()[0].id(),
            second_candidate.kv_state().kv_blocks()[0].id());
  EXPECT_EQ(first_candidate.kv_state().kv_blocks()[0].ref_count(), 2u);

  Sequence blocker =
      make_sequence(/*index=*/2, make_tokens(/*first=*/100, /*count=*/4));
  ASSERT_TRUE(pool.allocate(&blocker));

  const std::vector<size_t> prefix_blocks = pool.num_blocks_in_prefix_cache();
  ASSERT_EQ(prefix_blocks.size(), 1);
  EXPECT_EQ(prefix_blocks[0], 0);
  const std::vector<size_t> free_blocks = pool.num_free_blocks();
  ASSERT_EQ(free_blocks.size(), 1);
  EXPECT_EQ(free_blocks[0], 0);

  Sequence target =
      make_sequence(/*index=*/3, make_tokens(/*first=*/200, /*count=*/8));
  std::vector<Sequence*> release_candidates = {&first_candidate,
                                               &second_candidate};
  EXPECT_FALSE(pool.can_allocate_after_release(
      &target, target.num_tokens(), release_candidates))
      << "Both release candidates hold the same non-prefix-cache physical "
         "block, so scheduler planning must count it once, not once per "
         "candidate.";

  pool.deallocate_without_cache(&second_candidate);
  pool.deallocate_without_cache(&first_candidate);
  pool.deallocate_without_cache(&blocker);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       SingleGroupReleaseCandidatesSkipPartialSharedPrefixBlocks) {
  BlockManagerPool::Options options;
  options.num_blocks(5).block_size(4).enable_prefix_cache(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seed_sequence =
      make_sequence(/*index=*/0, make_tokens(/*first=*/1, /*count=*/8));
  const bool seed_allocated = pool.allocate(&seed_sequence);
  EXPECT_TRUE(seed_allocated);
  if (seed_allocated) {
    seed_sequence.kv_state().set_kv_cache_tokens_num(
        seed_sequence.num_tokens());
    pool.cache(&seed_sequence);
  }

  Sequence first_candidate =
      make_sequence(/*index=*/1,
                    make_tokens_with_suffix(/*prefix_first=*/1,
                                            /*prefix_count=*/8,
                                            /*suffix_first=*/100,
                                            /*suffix_count=*/4));
  const bool first_allocated = pool.allocate(&first_candidate);
  EXPECT_TRUE(first_allocated);

  Sequence second_candidate =
      make_sequence(/*index=*/2,
                    make_tokens_with_suffix(/*prefix_first=*/1,
                                            /*prefix_count=*/8,
                                            /*suffix_first=*/200,
                                            /*suffix_count=*/4));
  const bool second_allocated = pool.allocate(&second_candidate);
  EXPECT_TRUE(second_allocated);

  if (seed_allocated && first_allocated && second_allocated) {
    const bool shared_shape_ok =
        first_candidate.kv_state().num_kv_blocks() == 3 &&
        second_candidate.kv_state().num_kv_blocks() == 3 &&
        first_candidate.kv_state().shared_kv_blocks_num() == 2 &&
        second_candidate.kv_state().shared_kv_blocks_num() == 2;
    EXPECT_TRUE(shared_shape_ok);
    if (shared_shape_ok) {
      EXPECT_EQ(first_candidate.kv_state().kv_blocks()[0].id(),
                second_candidate.kv_state().kv_blocks()[0].id());
      EXPECT_EQ(first_candidate.kv_state().kv_blocks()[1].id(),
                second_candidate.kv_state().kv_blocks()[1].id());
      EXPECT_NE(first_candidate.kv_state().kv_blocks()[2].id(),
                second_candidate.kv_state().kv_blocks()[2].id());
      EXPECT_GT(first_candidate.kv_state().kv_blocks()[0].ref_count(), 3u);
      EXPECT_GT(first_candidate.kv_state().kv_blocks()[1].ref_count(), 3u);
      EXPECT_LE(first_candidate.kv_state().kv_blocks()[2].ref_count(), 2u);
      EXPECT_LE(second_candidate.kv_state().kv_blocks()[2].ref_count(), 2u);
    }

    const std::vector<size_t> free_blocks = pool.num_free_blocks();
    EXPECT_EQ(free_blocks.size(), 1);
    const bool free_shape_ok = free_blocks.size() == 1 && free_blocks[0] == 0;
    EXPECT_TRUE(free_shape_ok);
    if (shared_shape_ok && free_shape_ok) {
      Sequence target =
          make_sequence(/*index=*/3, make_tokens(/*first=*/300, /*count=*/12));
      std::vector<Sequence*> release_candidates = {&first_candidate,
                                                   &second_candidate};
      EXPECT_FALSE(pool.can_allocate_after_release(
          &target, target.num_tokens(), release_candidates))
          << "The two candidates only release their distinct suffix blocks. "
             "Their shared prefix blocks are still held by seed_sequence and "
             "the prefix cache, so they must not be counted toward a "
             "three-block target.";
    }
  }

  if (second_candidate.dp_rank() >= 0) {
    pool.deallocate_without_cache(&second_candidate);
  }
  if (first_candidate.dp_rank() >= 0) {
    pool.deallocate_without_cache(&first_candidate);
  }
  if (seed_sequence.dp_rank() >= 0) {
    pool.deallocate_without_cache(&seed_sequence);
  }

  int32_t dp_rank = -1;
  const size_t cleanup_tokens = static_cast<size_t>(options.num_blocks() - 1) *
                                static_cast<size_t>(options.block_size());
  std::vector<Block> cleanup_blocks = pool.allocate(cleanup_tokens, dp_rank);
  EXPECT_EQ(cleanup_blocks.size(), options.num_blocks() - 1);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       SingleGroupReleaseCapacityStaysWithinTargetDpRank) {
  BlockManagerPool::Options options;
  options.num_blocks(4).block_size(4).enable_prefix_cache(false);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence target_rank_blocker =
      make_sequence(/*index=*/0, make_tokens(/*first=*/1, /*count=*/12));
  target_rank_blocker.set_dp_rank(/*dp_rank=*/0);
  ASSERT_TRUE(pool.allocate(&target_rank_blocker));

  Sequence other_rank_candidate =
      make_sequence(/*index=*/1, make_tokens(/*first=*/100, /*count=*/8));
  other_rank_candidate.set_dp_rank(/*dp_rank=*/1);
  ASSERT_TRUE(pool.allocate(&other_rank_candidate));

  Sequence target =
      make_sequence(/*index=*/2, make_tokens(/*first=*/200, /*count=*/8));
  target.set_dp_rank(/*dp_rank=*/0);
  std::vector<Sequence*> release_candidates = {&other_rank_candidate};

  EXPECT_FALSE(pool.can_allocate_after_release(
      &target, target.num_tokens(), release_candidates))
      << "A target already bound to dp rank 0 must not borrow release capacity "
         "from candidates on dp rank 1.";

  pool.deallocate_without_cache(&other_rank_candidate);
  pool.deallocate_without_cache(&target_rank_blocker);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       SingleGroupUnboundTargetDoesNotMergeDpRankReleaseCapacity) {
  BlockManagerPool::Options options;
  options.num_blocks(4).block_size(4).enable_prefix_cache(false);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence rank0_candidate =
      make_sequence(/*index=*/0, make_tokens(/*first=*/1, /*count=*/8));
  rank0_candidate.set_dp_rank(/*dp_rank=*/0);
  ASSERT_TRUE(pool.allocate(&rank0_candidate));
  Sequence rank0_blocker =
      make_sequence(/*index=*/1, make_tokens(/*first=*/100, /*count=*/4));
  rank0_blocker.set_dp_rank(/*dp_rank=*/0);
  ASSERT_TRUE(pool.allocate(&rank0_blocker));

  Sequence rank1_candidate =
      make_sequence(/*index=*/2, make_tokens(/*first=*/200, /*count=*/8));
  rank1_candidate.set_dp_rank(/*dp_rank=*/1);
  ASSERT_TRUE(pool.allocate(&rank1_candidate));
  Sequence rank1_blocker =
      make_sequence(/*index=*/3, make_tokens(/*first=*/300, /*count=*/4));
  rank1_blocker.set_dp_rank(/*dp_rank=*/1);
  ASSERT_TRUE(pool.allocate(&rank1_blocker));

  Sequence target =
      make_sequence(/*index=*/4, make_tokens(/*first=*/400, /*count=*/12));
  std::vector<Sequence*> release_candidates = {&rank0_candidate,
                                               &rank1_candidate};

  EXPECT_FALSE(pool.can_allocate_after_release(
      &target, target.num_tokens(), release_candidates))
      << "An unbound target may choose a dp rank, but release capacity from "
         "different dp ranks must not be summed into one allocation.";

  pool.deallocate_without_cache(&rank1_blocker);
  pool.deallocate_without_cache(&rank1_candidate);
  pool.deallocate_without_cache(&rank0_blocker);
  pool.deallocate_without_cache(&rank0_candidate);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       SingleGroupBeamCandidatesDoNotCountSourceBlocksAsPrefixCache) {
  BlockManagerPool::Options options;
  options.num_blocks(5).block_size(4).enable_prefix_cache(true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence source =
      make_sequence(/*index=*/0, make_tokens(/*first=*/1, /*count=*/8));
  ASSERT_TRUE(pool.allocate(&source));

  Sequence first_candidate =
      make_sequence(/*index=*/1, make_tokens(/*first=*/100, /*count=*/12));
  Sequence second_candidate =
      make_sequence(/*index=*/2, make_tokens(/*first=*/200, /*count=*/12));
  first_candidate.set_dp_rank(source.dp_rank());
  second_candidate.set_dp_rank(source.dp_rank());
  {
    std::vector<Block> source_blocks = source.kv_state().kv_blocks();
    first_candidate.add_kv_blocks(source_blocks);
    second_candidate.add_kv_blocks(source_blocks);
  }
  ASSERT_TRUE(pool.allocate(&first_candidate, first_candidate.num_tokens()));
  ASSERT_TRUE(pool.allocate(&second_candidate, second_candidate.num_tokens()));

  ASSERT_EQ(first_candidate.kv_state().num_kv_blocks(), 3);
  ASSERT_EQ(second_candidate.kv_state().num_kv_blocks(), 3);
  EXPECT_EQ(first_candidate.kv_state().kv_blocks()[0].id(),
            second_candidate.kv_state().kv_blocks()[0].id());
  EXPECT_EQ(first_candidate.kv_state().kv_blocks()[1].id(),
            second_candidate.kv_state().kv_blocks()[1].id());
  EXPECT_NE(first_candidate.kv_state().kv_blocks()[2].id(),
            second_candidate.kv_state().kv_blocks()[2].id());
  EXPECT_EQ(first_candidate.kv_state().kv_blocks()[0].ref_count(), 3u);
  EXPECT_EQ(first_candidate.kv_state().kv_blocks()[1].ref_count(), 3u);
  EXPECT_EQ(first_candidate.kv_state().kv_blocks()[2].ref_count(), 1u);
  EXPECT_EQ(second_candidate.kv_state().kv_blocks()[2].ref_count(), 1u);

  const std::vector<size_t> prefix_blocks = pool.num_blocks_in_prefix_cache();
  ASSERT_EQ(prefix_blocks.size(), 1);
  EXPECT_EQ(prefix_blocks[0], 0);
  const std::vector<size_t> free_blocks = pool.num_free_blocks();
  ASSERT_EQ(free_blocks.size(), 1);
  EXPECT_EQ(free_blocks[0], 0);

  const std::vector<BlockGroupUsage> first_release =
      pool.estimate_release(&first_candidate);
  ASSERT_EQ(first_release.size(), 1);
  EXPECT_EQ(first_release[0].releasable_blocks, 1);

  Sequence target =
      make_sequence(/*index=*/3, make_tokens(/*first=*/300, /*count=*/16));
  std::vector<Sequence*> release_candidates = {&first_candidate,
                                               &second_candidate};
  EXPECT_FALSE(pool.can_allocate_after_release(
      &target, target.num_tokens(), release_candidates))
      << "The shared source KV blocks are held by a non-candidate sequence. "
         "That extra reference is not a prefix-cache reference, so only the "
         "two unique suffix blocks are actually releasable.";

  pool.deallocate_without_cache(&second_candidate);
  pool.deallocate_without_cache(&first_candidate);
  pool.deallocate_without_cache(&source);
}

TEST_F(CompositeSequenceBlockAllocatorTest,
       CompositeCanAllocateAfterReleaseSumsGroupEstimates) {
  BlockManagerPool::Options options;
  options.num_blocks(1)
      .block_size(1)
      .enable_prefix_cache(false)
      .composite_block_plan(make_composite_sum_plan());
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence first_candidate =
      make_sequence(/*index=*/0, make_tokens(/*first=*/1, /*count=*/8));
  ASSERT_TRUE(pool.allocate(&first_candidate));
  Sequence second_candidate =
      make_sequence(/*index=*/1, make_tokens(/*first=*/100, /*count=*/8));
  ASSERT_TRUE(pool.allocate(&second_candidate));

  Sequence target =
      make_sequence(/*index=*/2, make_tokens(/*first=*/200, /*count=*/12));
  std::vector<Sequence*> release_candidates = {&first_candidate,
                                               &second_candidate};

  EXPECT_TRUE(pool.can_allocate_after_release(
      &target, target.num_tokens(), release_candidates))
      << "Composite planning must sum releasable capacity per group through "
         "the composite allocator path. If it falls back to single-group "
         "physical block accounting, composite kv_blocks are empty and this "
         "target is incorrectly rejected.";

  pool.deallocate(&second_candidate);
  pool.deallocate(&first_candidate);
}

}  // namespace
}  // namespace xllm
