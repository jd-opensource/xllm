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

#include <gtest/gtest.h>

#include <type_traits>
#include <utility>

#include "block_manager_impl.h"
#include "block_manager_pool.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/block/cache_group.h"
#include "framework/request/incremental_decoder.h"
#include "framework/request/sequence_kv_state.h"

namespace xllm {

namespace {

template <typename T>
class ScopedValue final {
 public:
  ScopedValue(T* target, T value) : target_(target), old_(*target) {
    *target_ = value;
  }
  ~ScopedValue() { *target_ = old_; }

  ScopedValue(const ScopedValue&) = delete;
  ScopedValue& operator=(const ScopedValue&) = delete;

 private:
  T* target_;
  T old_;
};

template <typename T, typename = void>
struct HasEnableLinearStateOption : std::false_type {};

template <typename T>
struct HasEnableLinearStateOption<
    T,
    std::void_t<decltype(std::declval<T&>().enable_linear_state(true)),
                decltype(std::declval<const T&>().enable_linear_state())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasSequenceSingleBlockApi : std::false_type {};

template <typename T>
struct HasSequenceSingleBlockApi<
    T,
    std::void_t<decltype(std::declval<const T&>().has_single_block_id()),
                decltype(std::declval<const T&>().get_single_block_id()),
                decltype(std::declval<T&>().reset_single_block())>>
    : std::true_type {};

template <typename OptionsT>
bool EnableLinearStateOrFail(OptionsT& options) {
  if constexpr (HasEnableLinearStateOption<OptionsT>::value) {
    options.enable_linear_state(true);
    return true;
  }
  ADD_FAILURE() << "Task 2 missing APIs: BlockManagerPool::Options "
                   "enable_linear_state";
  return false;
}

template <typename SeqT>
bool HasSingleBlockIdOrFail(const SeqT& seq) {
  if constexpr (HasSequenceSingleBlockApi<SeqT>::value) {
    return seq.has_single_block_id();
  }
  ADD_FAILURE() << "Missing APIs: Sequence single-block handle";
  return false;
}

template <typename SeqT>
int32_t GetSingleBlockIdOrFail(const SeqT& seq) {
  if constexpr (HasSequenceSingleBlockApi<SeqT>::value) {
    return seq.get_single_block_id();
  }
  ADD_FAILURE() << "Missing APIs: Sequence single-block handle";
  return -1;
}

Sequence MakeSequence(size_t index, const std::vector<int32_t>& prompt_tokens) {
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;

  StoppingChecker stopping_checker;

  SequenceParams params;
  params.seq_capacity = prompt_tokens.size() + 8;
  params.echo = false;
  params.skip_special_tokens = true;
  params.streaming = false;
  params.enable_schedule_overlap = false;
  params.rec_type = RecType::kNone;
  params.bos_token_id = 0;
  params.request_id = "block_manager_pool_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;

  IncrementalDecoder decoder(
      /*prompt=*/"prompt",
      /*num_prompt_tokens=*/prompt_tokens.size(),
      /*echo=*/params.echo,
      /*skip_special_tokens=*/params.skip_special_tokens);

  return Sequence(index,
                  prompt_tokens,
                  /*input_embedding=*/torch::Tensor(),
                  /*mm_data=*/MMData(),
                  decoder,
                  params);
}

}  // namespace

TEST(BlockManagerTest, Basic) {
  const uint32_t n_blocks = 10;
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);
  EXPECT_EQ(manager.block_size(), block_size);

  // Allocate a block
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.id(), 1);
    EXPECT_EQ(block.size(), block_size);
    EXPECT_EQ(block.is_shared(), false);
    EXPECT_EQ(block.ref_count(), 1);

    EXPECT_EQ(manager.num_free_blocks(), n_blocks - 2);
  }
  // the block should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Allocate a list of blocks
  {
    std::vector<Block> blocks;
    for (uint32_t i = 1; i < n_blocks; ++i) {
      auto block = manager.allocate();
      EXPECT_EQ(block.id(), i);
      EXPECT_EQ(block.size(), block_size);
      EXPECT_EQ(block.is_shared(), false);
      EXPECT_EQ(block.ref_count(), 1);
      blocks.push_back(std::move(block));
    }
    EXPECT_EQ(manager.num_free_blocks(), 0);
    for (const auto& block : blocks) {
      EXPECT_EQ(block.ref_count(), 1);
      EXPECT_EQ(block.is_shared(), false);
    }

    // Test CHECK failure
    EXPECT_DEATH(manager.allocate(), "No more blocks available");
  }

  // all blocks should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Test shared blocks
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);
    // test copy constructor
    {
      // NOLINTNEXTLINE
      const Block block2 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block2.ref_count(), 2);
      EXPECT_EQ(block2.is_shared(), true);
      EXPECT_EQ(block2, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test assignment operator
    {
      Block block4 = manager.allocate();
      block4 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block4.ref_count(), 2);
      EXPECT_EQ(block4.is_shared(), true);
      EXPECT_EQ(block4, block);

      Block invalid_block;
      invalid_block = block;
      EXPECT_EQ(block.ref_count(), 3);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(invalid_block.ref_count(), 3);
      EXPECT_EQ(invalid_block.is_shared(), true);
      EXPECT_EQ(invalid_block, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test move constructor
    {
      Block block3 = std::move(block);
      EXPECT_FALSE(block.is_valid());

      EXPECT_EQ(block3.ref_count(), 1);
      EXPECT_EQ(block3.is_shared(), false);
      EXPECT_FALSE(block3 == block);
    }
    EXPECT_FALSE(block.is_valid());
  }
}

TEST(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenEnabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq));
  // id 0 is the reserved padding slot, so a real assignment is strictly
  // positive.
  EXPECT_GT(GetSingleBlockIdOrFail(seq), 0);
}

TEST(BlockManagerPoolTest, DeallocateReleasesSingleBlockId) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq1 = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&seq1));
  const int32_t id1 = GetSingleBlockIdOrFail(seq1);
  pool.deallocate(&seq1);
  EXPECT_FALSE(HasSingleBlockIdOrFail(seq1));

  Sequence seq2 = MakeSequence(1, /*prompt_tokens=*/{4, 5, 6});
  ASSERT_TRUE(pool.allocate(&seq2));
  EXPECT_EQ(GetSingleBlockIdOrFail(seq2), id1);
}

TEST(BlockManagerPoolTest, SingleBlockCapacityUsesOptionsMaxSeqs) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .enable_prefix_cache(false)
      .max_seqs_per_batch(4);
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::vector<Sequence> sequences;
  sequences.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    sequences.emplace_back(MakeSequence(i, /*prompt_tokens=*/{1}));
    EXPECT_TRUE(pool.allocate(&sequences.back()));
    EXPECT_TRUE(HasSingleBlockIdOrFail(sequences.back()));
  }
}

TEST(BlockManagerPoolTest, TryAllocateKvFailureRollsBackSingleBlock) {
  // unified scheduler-side single-block pool has 2 ids.
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(3).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  // This sequence needs far more KV blocks than available, forcing KV failure
  // after embedding and linear ids are allocated.
  std::vector<int32_t> huge_prompt(100, 1);
  Sequence fail_seq = MakeSequence(0, huge_prompt);
  EXPECT_FALSE(pool.try_allocate(&fail_seq));
  EXPECT_FALSE(HasSingleBlockIdOrFail(fail_seq));

  // The unified slot must have been rolled back, leaving enough capacity for
  // two new sequences to allocate.
  Sequence seq1 = MakeSequence(1, /*prompt_tokens=*/{1});
  Sequence seq2 = MakeSequence(2, /*prompt_tokens=*/{2});
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq1));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq2));
}

TEST(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenLinearStateDisabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = MakeSequence(0, /*prompt_tokens=*/{1, 2});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq));
}

TEST(BlockManagerPoolTest, SequenceCopyDoesNotReuseSingleBlockSlot) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  ASSERT_TRUE(EnableLinearStateOrFail(options));
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence src = MakeSequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&src));
  ASSERT_TRUE(HasSingleBlockIdOrFail(src));

  Sequence clone(src);
  EXPECT_FALSE(HasSingleBlockIdOrFail(clone));
  EXPECT_EQ(clone.get_single_block_id(), -1);

  ASSERT_TRUE(pool.allocate(&clone));
  EXPECT_TRUE(HasSingleBlockIdOrFail(clone));
  EXPECT_NE(GetSingleBlockIdOrFail(clone), GetSingleBlockIdOrFail(src));
}

// --- Composite-path BlockManagerPool (block-manager refactor) ---
// The normal-model pool now routes through ConcurrentCompositeBlockManager (a
// single C1 group) instead of the legacy BlockManagerImpl. These tests pin the
// integration the pool owns on top of the per-manager unit tests: path routing,
// allocate/deallocate accounting (with the prefix-cache insert now internal to
// the composite), and the cross-sequence prefix match plus whole-prompt
// back-off that composite_match_shared performs.

namespace {

// Three full C1 blocks worth of distinct, block-aligned prompt tokens.
const std::vector<int32_t>& CompositePrompt() {
  static const std::vector<int32_t> prompt = {
      11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34};
  return prompt;
}

BlockManagerPool::Options CompositeOptions(uint32_t num_blocks) {
  BlockManagerPool::Options options;
  options.num_blocks(num_blocks)
      .host_num_blocks(0)
      .block_size(4)
      .enable_prefix_cache(true)
      .num_single_blocks(8);
  return options;
}

}  // namespace

TEST(BlockManagerPoolCompositeTest, NormalModelRoutesThroughComposite) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool pool(CompositeOptions(/*num_blocks=*/16), /*dp_size=*/1);

  // One composite C1 manager per DP rank; its leaf reserves block 0 as padding,
  // so the scheduler-visible free count is num_blocks - 1 and nothing is used.
  ASSERT_EQ(pool.num_free_blocks().size(), 1u);
  EXPECT_EQ(pool.num_free_blocks()[0], 15u);
  EXPECT_EQ(pool.num_used_blocks()[0], 0u);
  EXPECT_EQ(pool.num_blocks_in_prefix_cache()[0], 0u);
}

TEST(BlockManagerPoolCompositeTest, AllocateFlushDeallocateAccounting) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool pool(CompositeOptions(/*num_blocks=*/32), /*dp_size=*/1);

  const size_t baseline_free = pool.num_free_blocks()[0];
  ASSERT_EQ(baseline_free, 31u);

  Sequence seq = MakeSequence(0, CompositePrompt());
  ASSERT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq));
  EXPECT_EQ(pool.num_free_blocks()[0], baseline_free - 3);
  EXPECT_EQ(pool.num_used_blocks()[0], 3u);

  // Commit the whole prompt so deallocate's internal insert caches all three
  // blocks before releasing them.
  seq.kv_state().set_kv_cache_tokens_num(seq.num_tokens());
  pool.deallocate(&seq);

  // The leaf marks the blocks logically released (used -> 0), but the group
  // prefix cache still pins their physical ids, so they never re-enter the free
  // list: three blocks live in the cache and free stays at baseline - 3.
  EXPECT_EQ(pool.num_used_blocks()[0], 0u);
  EXPECT_EQ(pool.num_free_blocks()[0], baseline_free - 3);
  EXPECT_EQ(pool.num_blocks_in_prefix_cache()[0], 3u);
}

TEST(BlockManagerPoolCompositeTest, PrefixMatchSeedsCachedTokensWithBackoff) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool pool(CompositeOptions(/*num_blocks=*/32), /*dp_size=*/1);

  // Sequence A allocates, commits, and on deallocate flushes three blocks into
  // the C1 prefix cache.
  Sequence seq_a = MakeSequence(0, CompositePrompt());
  ASSERT_TRUE(pool.allocate(&seq_a));
  seq_a.kv_state().set_kv_cache_tokens_num(seq_a.num_tokens());
  pool.deallocate(&seq_a);
  ASSERT_EQ(pool.num_blocks_in_prefix_cache()[0], 3u);

  // Sequence B re-issues the identical prompt. match restores all three blocks,
  // but because the whole prompt is cached the pool drops the last shared block
  // so the forward pass keeps at least one token to recompute:
  // shared_blocks_num
  // == 2 and cached tokens == 2 * block_size.
  Sequence seq_b = MakeSequence(1, CompositePrompt());
  ASSERT_TRUE(pool.allocate(&seq_b));

  EXPECT_EQ(seq_b.kv_state().kv_cache_tokens_num(), 8u);
  CacheGroupState* c1 = seq_b.kv_state().group_state(CacheStateId::C1);
  ASSERT_NE(c1, nullptr);
  EXPECT_EQ(c1->shared_blocks_num, 2u);
  // allocate() grows the group back to the full three blocks (two shared + one
  // freshly recomputed).
  EXPECT_EQ(c1->blocks.size(), 3u);
}

TEST(BlockManagerPoolCompositeTest, TryAllocateReservesWholePrompt) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool pool(CompositeOptions(/*num_blocks=*/32), /*dp_size=*/1);

  Sequence seq = MakeSequence(0, CompositePrompt());
  ASSERT_TRUE(pool.try_allocate(&seq));
  EXPECT_TRUE(HasSingleBlockIdOrFail(seq));
  // try_allocate reserves the whole prompt and marks every token cached.
  EXPECT_EQ(seq.kv_state().kv_cache_tokens_num(), seq.num_tokens());
  EXPECT_EQ(pool.num_used_blocks()[0], 3u);
}

TEST(BlockManagerPoolCompositeTest,
     LazyFlushInsertsCompletedBlocksOnNextAllocate) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool pool(CompositeOptions(/*num_blocks=*/32), /*dp_size=*/1);

  // Five block-aligned prompt tokens, allocated in two chunked-prefill steps.
  const std::vector<int32_t> prompt = {11, 12, 13, 14, 21, 22, 23, 24, 31, 32,
                                       33, 34, 41, 42, 43, 44, 51, 52, 53, 54};
  Sequence seq = MakeSequence(0, prompt);

  // First chunk: allocate three blocks. Nothing is committed, so nothing
  // flushes and the prefix cache stays empty -- even though no cache() call was
  // made.
  ASSERT_TRUE(pool.allocate(&seq, /*num_tokens=*/12));
  ASSERT_EQ(pool.num_used_blocks()[0], 3u);
  EXPECT_EQ(pool.num_blocks_in_prefix_cache()[0], 0u);

  // The forward commits the first three blocks.
  seq.kv_state().set_kv_cache_tokens_num(12);

  // Second chunk grows to five blocks. The lazy flush before this allocate
  // inserts the three completed blocks WITHOUT any scheduler cache() call,
  // proving the composite self-flushes on grow.
  ASSERT_TRUE(pool.allocate(&seq, /*num_tokens=*/20));
  EXPECT_EQ(pool.num_used_blocks()[0], 5u);
  EXPECT_EQ(pool.num_blocks_in_prefix_cache()[0], 3u);
}

TEST(BlockManagerPoolCompositeTest, DeallocateWithoutCacheReleasesEverything) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool pool(CompositeOptions(/*num_blocks=*/32), /*dp_size=*/1);

  const size_t baseline_free = pool.num_free_blocks()[0];
  Sequence seq = MakeSequence(0, CompositePrompt());
  ASSERT_TRUE(pool.allocate(&seq));
  ASSERT_EQ(pool.num_used_blocks()[0], 3u);

  // No flush: every block returns to the free list and nothing is cached.
  pool.deallocate_without_cache(&seq);
  EXPECT_EQ(pool.num_used_blocks()[0], 0u);
  EXPECT_EQ(pool.num_free_blocks()[0], baseline_free);
  EXPECT_EQ(pool.num_blocks_in_prefix_cache()[0], 0u);
}

}  // namespace xllm
