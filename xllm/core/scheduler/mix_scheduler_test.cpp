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

#include "mix_scheduler.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "common/global_flags.h"
#include "distributed_runtime/engine.h"

namespace xllm {

namespace {

class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens = true) const {
    NOT_IMPLEMENTED();
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const {
    NOT_IMPLEMENTED();
  }
  std::optional<int32_t> token_to_id(const std::string_view& token) const {
    NOT_IMPLEMENTED();
  }
  std::string id_to_token(int32_t id) const { NOT_IMPLEMENTED(); }
  size_t vocab_size() const { NOT_IMPLEMENTED(); }
  std::unique_ptr<Tokenizer> clone() const {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine : public Engine {
 public:
  FakeEngine(int32_t num_blocks, int32_t block_size) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.enable_prefix_cache_ = true;
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ = std::make_unique<BlockManagerPool>(opt, 1);
  }
  ForwardOutput step(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  void update_last_step_result(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  const Tokenizer* tokenizer() const { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const { NOT_IMPLEMENTED(); }
  const TokenizerArgs& tokenizer_args() const { NOT_IMPLEMENTED(); }
  std::vector<int64_t> get_active_activation_memory() const {
    NOT_IMPLEMENTED();
  }
  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<BlockManagerPool> fake_block_manager_;
};

ContinuousScheduler::Options create_scheduler_options(
    int32_t max_tokens_per_batch,
    int32_t max_seqs_per_batch,
    int32_t num_speculative_tokens,
    int32_t max_tokens_per_chunk_for_prefill,
    int32_t dp_size) {
  ContinuousScheduler::Options opt;
  opt.num_speculative_tokens_ = num_speculative_tokens;
  opt.max_tokens_per_chunk_for_prefill_ = max_tokens_per_chunk_for_prefill;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.priority_strategy_ = "fcfs";
  opt.enable_profile_kv_blocks_ = true;
  opt.enable_latency_aware_schedule_ = false;
  opt.max_global_ttft_ms_ = std::numeric_limits<int32_t>::max();
  opt.max_global_tpot_ms_ = std::numeric_limits<int32_t>::max();
  return opt;
}

std::shared_ptr<Request> generate_request(const std::vector<int32_t>& prompt,
                                          int32_t max_tokens) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(max_tokens);
  stopping_checker.set_max_context_len(static_cast<int32_t>(prompt.size()) +
                                       max_tokens + 128);
  stopping_checker.set_ignore_eos(true);

  RequestState req_state("x",
                         prompt,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         static_cast<int32_t>(prompt.size()) + max_tokens + 256,
                         1,
                         1,
                         false,
                         false,
                         false,
                         false,
                         false,
                         nullptr,
                         nullptr);
  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

}  // namespace

TEST(MixSchedulerTest, ReusePrefixBlocksWithinBatch) {
  FLAGS_enable_prefix_cache = true;
  FLAGS_enable_in_batch_prefix_cache_reuse = true;

  const int32_t block_size = 16;
  const int32_t prompt_len = 128;

  auto engine = std::make_unique<FakeEngine>(64, block_size);
  auto options = create_scheduler_options(1024, 16, 0, 1024, 1);
  auto scheduler = std::make_unique<MixScheduler>(engine.get(), options);

  const std::vector<int32_t> prompt(prompt_len, 42);
  auto req0 = generate_request(prompt, 8);
  auto req1 = generate_request(prompt, 8);
  scheduler->add_request(req0);
  scheduler->add_request(req1);

  auto batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0].size(), 2);

  auto* seq0 = batches[0][0];
  auto* seq1 = batches[0][1];
  const auto& budgets = batches[0].get_allowed_max_tokens();
  ASSERT_EQ(budgets.size(), 2);

  Sequence* producer = nullptr;
  Sequence* consumer = nullptr;
  uint32_t producer_budget = 0;
  uint32_t consumer_budget = 0;

  if (seq0->kv_state().kv_cache_tokens_num() >
      seq1->kv_state().kv_cache_tokens_num()) {
    consumer = seq0;
    producer = seq1;
    consumer_budget = budgets[0];
    producer_budget = budgets[1];
  } else {
    consumer = seq1;
    producer = seq0;
    consumer_budget = budgets[1];
    producer_budget = budgets[0];
  }

  ASSERT_NE(producer, nullptr);
  ASSERT_NE(consumer, nullptr);
  ASSERT_GT(consumer->kv_state().kv_cache_tokens_num(), 0);

  const size_t expected_shared_blocks = prompt_len / block_size - 1;
  EXPECT_EQ(consumer->kv_state().shared_kv_blocks_num(),
            expected_shared_blocks);
  EXPECT_EQ(consumer->kv_state().kv_cache_tokens_num(),
            expected_shared_blocks * block_size);
  EXPECT_EQ(consumer_budget, static_cast<uint32_t>(block_size));
  EXPECT_EQ(producer_budget, static_cast<uint32_t>(prompt_len));

  ASSERT_GE(producer->kv_state().num_kv_blocks(), expected_shared_blocks);
  ASSERT_GE(consumer->kv_state().num_kv_blocks(), expected_shared_blocks);
  for (size_t i = 0; i < expected_shared_blocks; ++i) {
    EXPECT_EQ(producer->kv_state().kv_blocks()[i].id(),
              consumer->kv_state().kv_blocks()[i].id());
  }
}

TEST(MixSchedulerTest, IdenticalRequestsStayDeterministicWithoutSampling) {
  FLAGS_enable_prefix_cache = true;
  FLAGS_enable_in_batch_prefix_cache_reuse = true;

  const int32_t block_size = 16;
  const int32_t prompt_len = 128;

  auto engine = std::make_unique<FakeEngine>(64, block_size);
  auto options = create_scheduler_options(1024, 16, 0, 1024, 1);
  auto scheduler = std::make_unique<MixScheduler>(engine.get(), options);

  const std::vector<int32_t> prompt(prompt_len, 7);
  auto req0 = generate_request(prompt, 8);
  auto req1 = generate_request(prompt, 8);
  scheduler->add_request(req0);
  scheduler->add_request(req1);

  auto batches = scheduler->prepare_batch_test();
  ASSERT_EQ(batches.size(), 1);
  ASSERT_EQ(batches[0].size(), 2);
  const auto& budgets = batches[0].get_allowed_max_tokens();
  ASSERT_EQ(budgets.size(), 2);

  auto* seq0 = batches[0][0];
  auto* seq1 = batches[0][1];

  // Same request content should keep exactly the same token stream.
  ASSERT_EQ(seq0->tokens(), seq1->tokens());
  ASSERT_EQ(seq0->num_prompt_tokens(), seq1->num_prompt_tokens());

  const auto* sp0 = seq0->sampling_param();
  const auto* sp1 = seq1->sampling_param();
  ASSERT_NE(sp0, nullptr);
  ASSERT_NE(sp1, nullptr);
  EXPECT_FALSE(sp0->do_sample);
  EXPECT_FALSE(sp1->do_sample);
  EXPECT_FLOAT_EQ(sp0->temperature, 0.0f);
  EXPECT_FLOAT_EQ(sp1->temperature, 0.0f);
  EXPECT_FLOAT_EQ(sp0->top_p, 1.0f);
  EXPECT_FLOAT_EQ(sp1->top_p, 1.0f);
  EXPECT_EQ(sp0->top_k, -1);
  EXPECT_EQ(sp1->top_k, -1);
  EXPECT_EQ(sp0->beam_width, 0);
  EXPECT_EQ(sp1->beam_width, 0);

  // Although one sequence may reuse in-batch prefix blocks, both sequences
  // should target the same logical prompt length for this prefill step.
  const size_t logical_prefill_target_0 =
      seq0->kv_state().kv_cache_tokens_num() + budgets[0];
  const size_t logical_prefill_target_1 =
      seq1->kv_state().kv_cache_tokens_num() + budgets[1];
  EXPECT_EQ(logical_prefill_target_0, seq0->num_prompt_tokens());
  EXPECT_EQ(logical_prefill_target_1, seq1->num_prompt_tokens());
  EXPECT_EQ(logical_prefill_target_0, logical_prefill_target_1);

  // Ensure batch-local prefix sharing is actually triggered in this scenario.
  const bool has_in_batch_shared_prefix =
      seq0->kv_state().shared_kv_blocks_num() > 0 ||
      seq1->kv_state().shared_kv_blocks_num() > 0;
  EXPECT_TRUE(has_in_batch_shared_prefix);
}

}  // namespace xllm
