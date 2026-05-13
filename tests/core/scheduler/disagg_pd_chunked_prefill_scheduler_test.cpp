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

namespace xllm {

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

}  // namespace xllm
