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

#include "framework/prefix_cache/in_batch_prefix_cache.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "framework/block/block_manager_impl.h"
#include "framework/prefix_cache/prefix_cache.h"
#include "framework/request/request.h"

namespace xllm {

namespace {

constexpr int32_t kBlockSize = 16;

std::vector<int32_t> make_block_prompt(
    const std::initializer_list<int32_t>& block_tokens) {
  std::vector<int32_t> prompt;
  prompt.reserve(block_tokens.size() * kBlockSize);
  for (int32_t token_id : block_tokens) {
    prompt.insert(prompt.end(), kBlockSize, token_id);
  }
  return prompt;
}

std::shared_ptr<Request> make_request(const std::vector<int32_t>& prompt,
                                      const std::string& request_id) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(8);
  stopping_checker.set_max_context_len(
      static_cast<int32_t>(prompt.size()) + 64);
  stopping_checker.set_ignore_eos(true);

  RequestState request_state("x",
                             prompt,
                             sampling_param,
                             scheduler_param,
                             stopping_checker,
                             prompt.size() + 128,
                             1,
                             1,
                             false,
                             false,
                             false,
                             false,
                             false,
                             nullptr,
                             nullptr);
  return std::make_shared<Request>(
      request_id, request_id, request_id, std::move(request_state), request_id);
}

Sequence* get_sequence(const std::shared_ptr<Request>& request) {
  return request->sequences().front().get();
}

void allocate_prompt_blocks(Sequence* sequence, BlockManagerImpl* manager) {
  ASSERT_NE(sequence, nullptr);
  ASSERT_NE(manager, nullptr);
  ASSERT_EQ(sequence->num_tokens() % kBlockSize, 0UL);
  sequence->add_kv_blocks(manager->allocate(sequence->num_tokens() / kBlockSize));
}

void seed_consumer_prefix(Sequence* sequence,
                          size_t cached_full_blocks,
                          BlockManagerImpl* manager) {
  ASSERT_NE(sequence, nullptr);
  ASSERT_NE(manager, nullptr);
  ASSERT_EQ(sequence->num_tokens() % kBlockSize, 0UL);
  ASSERT_GT(cached_full_blocks, 0UL);
  sequence->add_kv_blocks(manager->allocate(cached_full_blocks));
  auto* blocks = sequence->kv_state().mutable_kv_blocks();
  ASSERT_NE(blocks, nullptr);
  ASSERT_EQ(PrefixCache::compute_hash_keys(sequence->tokens(), *blocks),
            cached_full_blocks);
  sequence->kv_state().set_kv_cache_tokens_num(cached_full_blocks * kBlockSize);
}

}  // namespace

TEST(InBatchPrefixCacheContextTest, ChoosesLongestContinuousMatch) {
  BlockManager::Options options;
  options.num_blocks(64).block_size(kBlockSize);
  BlockManagerImpl block_manager(options);
  InBatchPrefixCacheContext context;

  auto short_provider_request =
      make_request(make_block_prompt({11, 22}), "short_provider");
  auto long_provider_request =
      make_request(make_block_prompt({11, 22, 33}), "long_provider");
  auto consumer_request =
      make_request(make_block_prompt({11, 22, 33, 44}), "consumer");

  Sequence* short_provider = get_sequence(short_provider_request);
  Sequence* long_provider = get_sequence(long_provider_request);
  Sequence* consumer = get_sequence(consumer_request);

  allocate_prompt_blocks(short_provider, &block_manager);
  allocate_prompt_blocks(long_provider, &block_manager);
  seed_consumer_prefix(consumer, /*cached_full_blocks=*/1, &block_manager);

  short_provider->set_dp_rank(0);
  long_provider->set_dp_rank(0);

  context.register_provider(
      short_provider, kBlockSize, short_provider->num_tokens());
  context.register_provider(long_provider, kBlockSize, long_provider->num_tokens());
  context.try_match(consumer, kBlockSize);

  ASSERT_EQ(consumer->dp_rank(), 0);
  ASSERT_EQ(consumer->kv_state().shared_kv_blocks_num(), 3UL);
  ASSERT_EQ(consumer->kv_state().kv_cache_tokens_num(), 3 * kBlockSize);
  ASSERT_EQ(consumer->kv_state().num_kv_blocks(), 3UL);

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(consumer->kv_state().kv_blocks()[i].id(),
              long_provider->kv_state().kv_blocks()[i].id());
  }
}

TEST(InBatchPrefixCacheContextTest, NoHitLeavesConsumerStateUntouched) {
  BlockManager::Options options;
  options.num_blocks(64).block_size(kBlockSize);
  BlockManagerImpl block_manager(options);
  InBatchPrefixCacheContext context;

  auto provider_request =
      make_request(make_block_prompt({11, 99}), "provider");
  auto consumer_request =
      make_request(make_block_prompt({11, 22, 33}), "consumer");

  Sequence* provider = get_sequence(provider_request);
  Sequence* consumer = get_sequence(consumer_request);

  allocate_prompt_blocks(provider, &block_manager);
  seed_consumer_prefix(consumer, /*cached_full_blocks=*/1, &block_manager);
  provider->set_dp_rank(0);

  const auto original_block_id = consumer->kv_state().kv_blocks()[0].id();
  context.register_provider(provider, kBlockSize, provider->num_tokens());
  context.try_match(consumer, kBlockSize);

  EXPECT_EQ(consumer->dp_rank(), -1);
  ASSERT_EQ(consumer->kv_state().num_kv_blocks(), 1UL);
  EXPECT_EQ(consumer->kv_state().shared_kv_blocks_num(), 0UL);
  EXPECT_EQ(consumer->kv_state().kv_cache_tokens_num(), 1 * kBlockSize);
  EXPECT_EQ(consumer->kv_state().kv_blocks()[0].id(), original_block_id);
}

TEST(InBatchPrefixCacheContextTest, DoesNotMatchAcrossDpRanks) {
  BlockManager::Options options;
  options.num_blocks(64).block_size(kBlockSize);
  BlockManagerImpl block_manager(options);
  InBatchPrefixCacheContext context;

  auto provider_request =
      make_request(make_block_prompt({11, 22, 33}), "provider");
  auto consumer_request =
      make_request(make_block_prompt({11, 22, 33, 44}), "consumer");

  Sequence* provider = get_sequence(provider_request);
  Sequence* consumer = get_sequence(consumer_request);

  allocate_prompt_blocks(provider, &block_manager);
  provider->set_dp_rank(0);
  consumer->set_dp_rank(1);

  context.register_provider(provider, kBlockSize, provider->num_tokens());
  context.try_match(consumer, kBlockSize);

  EXPECT_EQ(consumer->dp_rank(), 1);
  EXPECT_EQ(consumer->kv_state().shared_kv_blocks_num(), 0UL);
  EXPECT_EQ(consumer->kv_state().kv_cache_tokens_num(), 0UL);
  EXPECT_EQ(consumer->kv_state().num_kv_blocks(), 0UL);
}

}  // namespace xllm
