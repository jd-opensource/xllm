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

#include "scheduler/disagg_pd_chunked_prefill_scheduler.h"

#include <gtest/gtest.h>

#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/request/sequence.h"

namespace xllm {

namespace {

std::shared_ptr<Request> make_request_with_best_of(
    const std::vector<int32_t>& prompt_token_ids,
    size_t n,
    size_t best_of) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(8);
  stopping_checker.set_max_context_len(30000);
  stopping_checker.set_ignore_eos(true);

  RequestState req_state("x",
                         prompt_token_ids,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt_token_ids.size() + 30000,
                         n,
                         best_of,
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

TEST(DisaggPDChunkedPrefillSchedulerTest, PicksCurrentChunkBudget) {
  const PDChunkBudget budget = pick_pd_chunk_budget(32, 96, 40, 64);
  EXPECT_EQ(budget.next_tokens, 40);
  EXPECT_EQ(budget.max_tokens, 72);
}

TEST(DisaggPDChunkedPrefillSchedulerTest, LastPromptChunkStopsAtPromptEnd) {
  const PDChunkBudget budget = pick_pd_chunk_budget(80, 96, 40, 64);
  EXPECT_EQ(budget.next_tokens, 16);
  EXPECT_EQ(budget.max_tokens, 96);
}

TEST(DisaggPDChunkedPrefillSchedulerTest, EmptyBudgetRejectsSchedule) {
  const PDChunkBudget budget = pick_pd_chunk_budget(32, 96, 40, 0);
  EXPECT_EQ(budget.next_tokens, 0);
  EXPECT_EQ(budget.max_tokens, 32);
}

// Regression test: in disagg PD mode, expansion of best_of_n sequences must
// be deferred to the DECODE instance (where prefix cache lets seq[1..N-1]
// share seq[0]'s prompt KV). Expanding on the PREFILL instance would waste
// N x prefill compute. expand_sequences(false) is still the API used in
// non-PD ChunkedPrefillScheduler, so this test pins the contract: a fresh
// request created with best_of=4 starts with exactly one sequence, and the
// PD-PREFILL prepare_batch is expected NOT to call expand_sequences(false).
TEST(DisaggPDChunkedPrefillSchedulerTest,
     BestOfNRequestStartsWithSingleSequence) {
  auto request = make_request_with_best_of(
      {1, 2, 3, 4, 5, 6, 7, 8}, /*n=*/2, /*best_of=*/4);
  EXPECT_EQ(request->sequences().size(), 1u);
}

// Regression test: expand_sequences(true) should be a no-op while the first
// sequence has not finished prefill yet (kv_cache_tokens_num <
// num_prompt_tokens). This is the precondition that makes the two-phase
// flow on the decode instance correct: seq[1..best_of-1] are only created
// after seq[0]'s prompt KV is available for prefix-cache reuse.
TEST(DisaggPDChunkedPrefillSchedulerTest,
     ExpandSharePrefixWaitsForFirstSequencePrefill) {
  auto request = make_request_with_best_of(
      {1, 2, 3, 4, 5, 6, 7, 8}, /*n=*/2, /*best_of=*/4);
  EXPECT_FALSE(request->expand_sequences(/*share_prefix=*/true));
  EXPECT_EQ(request->sequences().size(), 1u);

  Sequence* seq0 = request->sequences()[0].get();
  seq0->kv_state().set_kv_cache_tokens_num(seq0->num_prompt_tokens());
  EXPECT_TRUE(request->expand_sequences(/*share_prefix=*/true));
  EXPECT_EQ(request->sequences().size(), 4u);
}

}  // namespace xllm
