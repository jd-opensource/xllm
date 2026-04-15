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

#include "rec_sampler.h"

#include <gtest/gtest.h>

#include "core/common/global_flags.h"

namespace xllm {

TEST(RecSamplerTest, MultiRoundFastPathUsesFullLogSoftmaxScores) {
  const bool old_enable_rec_fast_sampler = FLAGS_enable_rec_fast_sampler;
  const bool old_enable_qwen3_reranker = FLAGS_enable_qwen3_reranker;
  const int32_t old_max_decode_rounds = FLAGS_max_decode_rounds;
  FLAGS_enable_rec_fast_sampler = true;
  FLAGS_enable_qwen3_reranker = false;
  FLAGS_max_decode_rounds = 2;

  RecSampler sampler(RecPipelineType::kLlmRecMultiRoundPipeline);

  SamplingParameters params;
  params.selected_token_idxes = torch::tensor({0, 1}, torch::kInt32);
  params.sample_idxes = torch::tensor({0, 1}, torch::kInt32);
  params.do_sample = torch::tensor({true, true}, torch::kBool);
  params.all_random_sample = true;
  params.all_greedy_sample = false;
  params.logprobs = true;
  params.max_top_logprobs = 2;
  params.use_beam_search = true;

  torch::Tensor logits = torch::tensor(
      {{1.0f, 0.5f, -2.0f}, {0.2f, 0.1f, 0.0f}}, torch::dtype(torch::kFloat32));

  SampleOutput output = sampler.forward(logits, params);

  torch::Tensor logprobs =
      torch::log_softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  auto [expected_top_logprobs, expected_top_tokens] =
      logprobs.topk(/*k=*/2, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);

  EXPECT_TRUE(torch::equal(output.top_tokens, expected_top_tokens));
  EXPECT_TRUE(torch::allclose(output.top_logprobs,
                              expected_top_logprobs,
                              /*rtol=*/1e-5,
                              /*atol=*/1e-6));
  EXPECT_TRUE(torch::equal(
      output.next_tokens,
      expected_top_tokens.select(/*dim=*/1, /*index=*/0).to(torch::kLong)));
  EXPECT_TRUE(torch::allclose(
      output.logprobs,
      expected_top_logprobs.select(/*dim=*/1, /*index=*/0).contiguous(),
      /*rtol=*/1e-5,
      /*atol=*/1e-6));

  FLAGS_enable_rec_fast_sampler = old_enable_rec_fast_sampler;
  FLAGS_enable_qwen3_reranker = old_enable_qwen3_reranker;
  FLAGS_max_decode_rounds = old_max_decode_rounds;
}

}  // namespace xllm
