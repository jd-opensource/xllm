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

#include "runtime/rec_beam_utils.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::runtime::detail {
namespace {

TEST(RecBeamUtilsTest, WriteFirstRoundBeamOutputsReshapesBatchBeamSlice) {
  constexpr int32_t kBatchSize = 2;
  constexpr int32_t kBeamWidth = 64;
  constexpr int32_t kTotalRounds = 3;
  const auto int_options = torch::TensorOptions().dtype(torch::kInt32);
  const auto fp32_options = torch::TensorOptions().dtype(torch::kFloat32);

  auto top_tokens =
      torch::arange(kBatchSize * kBeamWidth, int_options).view({-1, 1});
  auto top_logprobs =
      torch::arange(kBatchSize * kBeamWidth, fp32_options).view({-1, 1});
  auto out_token_ids = torch::zeros({kBatchSize * kBeamWidth, 1}, int_options);
  auto out_log_probs = torch::zeros({kBatchSize * kBeamWidth, 1}, fp32_options);
  auto out_seqgroup =
      torch::zeros({kBatchSize, kBeamWidth, kTotalRounds}, int_options);

  write_first_round_beam_outputs(top_tokens,
                                 top_logprobs,
                                 kBatchSize,
                                 out_token_ids,
                                 out_log_probs,
                                 out_seqgroup);

  EXPECT_TRUE(out_token_ids.equal(top_tokens));
  EXPECT_TRUE(out_log_probs.equal(top_logprobs));
  EXPECT_TRUE(out_seqgroup.select(/*dim=*/2, /*index=*/0)
                  .equal(top_tokens.view({kBatchSize, kBeamWidth})));
  EXPECT_EQ(out_seqgroup.select(/*dim=*/2, /*index=*/1).sum().item<int32_t>(),
            0);
  EXPECT_EQ(out_seqgroup.select(/*dim=*/2, /*index=*/2).sum().item<int32_t>(),
            0);
}

}  // namespace
}  // namespace xllm::runtime::detail
