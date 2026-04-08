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

#include "common/global_flags.h"

namespace xllm {

namespace {

SamplingParameters CreateBeamSamplingParameters() {
  RequestSamplingParam request_param;
  request_param.logprobs = true;
  request_param.top_logprobs = 2;
  request_param.beam_width = 2;

  SamplingParameters params;
  params.init(std::vector<const RequestSamplingParam*>{&request_param},
              std::vector<int32_t>{0},
              std::vector<int32_t>{0},
              std::vector<std::vector<int64_t>>{{}},
              std::vector<std::vector<int32_t>>{{}},
              std::vector<int32_t>{0});
  return params;
}

}  // namespace

TEST(RecSamplerTest, FastPathEnablementIsInstanceScoped) {
  FLAGS_enable_rec_fast_sampler = false;

  RecSampler llm_sampler(RecPipelineType::kLlmRecMultiRoundPipeline,
                         /*enable_fast_path=*/true);
  RecSampler rec_sampler(RecPipelineType::kLlmRecMultiRoundPipeline,
                         /*enable_fast_path=*/FLAGS_enable_rec_fast_sampler);

  EXPECT_TRUE(llm_sampler.fast_path_enabled());
  EXPECT_FALSE(rec_sampler.fast_path_enabled());
}

TEST(RecSamplerTest, ForwardResultIsStableAcrossFastPathToggleOnCpu) {
  auto logits = torch::tensor({{0.1F, 0.9F, 0.0F}});
  auto params = CreateBeamSamplingParameters();

  RecSampler fast_sampler(RecPipelineType::kLlmRecMultiRoundPipeline,
                          /*enable_fast_path=*/true);
  RecSampler fallback_sampler(RecPipelineType::kLlmRecMultiRoundPipeline,
                              /*enable_fast_path=*/false);

  auto fast_output = fast_sampler.forward(logits, params);
  auto fallback_output = fallback_sampler.forward(logits, params);

  EXPECT_TRUE(
      torch::equal(fast_output.next_tokens, fallback_output.next_tokens));
  EXPECT_TRUE(torch::equal(fast_output.top_tokens, fallback_output.top_tokens));
  EXPECT_TRUE(torch::allclose(
      fast_output.logprobs, fallback_output.logprobs, 1e-6, 1e-6));
  EXPECT_TRUE(torch::allclose(
      fast_output.top_logprobs, fallback_output.top_logprobs, 1e-6, 1e-6));
}

}  // namespace xllm
