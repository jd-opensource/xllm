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

#include "runtime/profile_run.h"

#include <gtest/gtest.h>

#include "framework/model/model_args.h"

namespace xllm {
namespace runtime {

TEST(ProfileRunTest, UseProfileRunOnlyOnMluNonPd) {
  Options opt;
  opt.enable_profile_run(true).enable_disagg_pd(false);
  EXPECT_TRUE(use_profile_run(opt, /*is_mlu_build=*/true));
  EXPECT_FALSE(use_profile_run(opt, /*is_mlu_build=*/false));
  opt.enable_disagg_pd(true);
  EXPECT_FALSE(use_profile_run(opt, /*is_mlu_build=*/true));
}

TEST(ProfileRunTest, PickProfileTokensUsesChunkCap) {
  Options opt;
  opt.enable_chunked_prefill(true)
      .max_tokens_per_batch(8192)
      .max_tokens_per_chunk_for_prefill(2048);
  EXPECT_EQ(pick_profile_tokens(opt), 2048);
}

TEST(ProfileRunTest, BuildPlanUsesCumulativeSeqLensOnMlu) {
  ModelArgs args;
  args.n_layers(2);
  args.n_heads(8);
  args.n_kv_heads(8);
  args.head_dim(128);

  Options opt;
  opt.max_seqs_per_batch(3)
      .max_tokens_per_batch(3072)
      .max_tokens_per_chunk_for_prefill(3072)
      .enable_chunked_prefill(false);

  ProfilePlan plan = build_profile_plan(args,
                                        opt,
                                        /*block_size=*/128,
                                        /*is_mla=*/false,
                                        /*is_mlu_build=*/true);
  EXPECT_EQ(plan.seq_lens, std::vector<int32_t>({0, 1024, 2048, 3072}));
  EXPECT_EQ(plan.block_tables[0],
            std::vector<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(plan.block_tables[1],
            std::vector<int32_t>({8, 9, 10, 11, 12, 13, 14, 15}));
  EXPECT_EQ(plan.block_tables[2],
            std::vector<int32_t>({16, 17, 18, 19, 20, 21, 22, 23}));
  EXPECT_EQ(plan.num_tokens, 3072);
}

TEST(ProfileRunTest, BuildPlanKeepsRuntimeLayout) {
  ModelArgs args;

  Options opt;
  opt.max_seqs_per_batch(2)
      .max_tokens_per_batch(9)
      .max_tokens_per_chunk_for_prefill(9)
      .block_size(4)
      .enable_chunked_prefill(false);

  ProfilePlan plan = build_profile_plan(args,
                                        opt,
                                        /*block_size=*/4,
                                        /*is_mla=*/false,
                                        /*is_mlu_build=*/true);
  EXPECT_EQ(plan.seq_lens, std::vector<int32_t>({0, 5, 9}));
  ASSERT_EQ(plan.block_tables.size(), 2);
  EXPECT_EQ(plan.block_tables[0], std::vector<int32_t>({0, 1}));
  EXPECT_EQ(plan.block_tables[1], std::vector<int32_t>({2}));
}

TEST(ProfileRunTest, CalcRuntimePeakUsesReservedDeltaAfterTmpKv) {
  PeakMem base;
  base.alloc_bytes = 100;
  base.cache_bytes = 140;
  PeakMem peak;
  peak.alloc_bytes = 150;
  peak.cache_bytes = 210;
  EXPECT_EQ(calc_runtime_peak(base, peak), 70);
}

TEST(ProfileRunTest, CalcRuntimePeakNeverReturnsNegative) {
  PeakMem base;
  base.alloc_bytes = 200;
  base.cache_bytes = 220;
  PeakMem peak;
  peak.alloc_bytes = 180;
  peak.cache_bytes = 210;
  EXPECT_EQ(calc_runtime_peak(base, peak), 0);
}

TEST(ProfileRunTest, CalcSafeKvBytesUsesMinAcrossWorkers) {
  Options opt;
  opt.max_memory_utilization(0.8).max_cache_size(0);

  ProfileMem worker_a;
  worker_a.total_bytes = 1000;
  worker_a.weight_bytes = 300;
  worker_a.runtime_peak_bytes = 100;
  worker_a.ok = true;

  ProfileMem worker_b;
  worker_b.total_bytes = 900;
  worker_b.weight_bytes = 250;
  worker_b.runtime_peak_bytes = 180;
  worker_b.ok = true;

  EXPECT_EQ(
      calc_safe_kv_bytes(std::vector<ProfileMem>{worker_a, worker_b}, opt),
      290);
}

TEST(ProfileRunTest, CalcSafeKvBytesRespectsMaxCacheSize) {
  Options opt;
  opt.max_memory_utilization(0.9).max_cache_size(128);

  ProfileMem worker_a;
  worker_a.total_bytes = 1000;
  worker_a.weight_bytes = 200;
  worker_a.runtime_peak_bytes = 100;
  worker_a.ok = true;

  EXPECT_EQ(calc_safe_kv_bytes(std::vector<ProfileMem>{worker_a}, opt), 128);
}

}  // namespace runtime
}  // namespace xllm
