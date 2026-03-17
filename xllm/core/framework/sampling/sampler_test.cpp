/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "sampler.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string>
#include <utility>
#include <vector>

#include "common/global_flags.h"
#include "sampler_utils.h"

namespace xllm {

namespace {

class Qwen3RerankerFlagGuard {
 public:
  Qwen3RerankerFlagGuard()
      : enable_qwen3_reranker_(FLAGS_enable_qwen3_reranker) {}

  ~Qwen3RerankerFlagGuard() {
    FLAGS_enable_qwen3_reranker = enable_qwen3_reranker_;
  }

 private:
  bool enable_qwen3_reranker_;
};

class CandidateFlagsGuard {
 public:
  CandidateFlagsGuard()
      : candidate_token_ids_(FLAGS_candidate_token_ids),
        enable_qwen3_reranker_(FLAGS_enable_qwen3_reranker) {}

  ~CandidateFlagsGuard() {
    FLAGS_candidate_token_ids = candidate_token_ids_;
    FLAGS_enable_qwen3_reranker = enable_qwen3_reranker_;
  }

 private:
  std::string candidate_token_ids_;
  bool enable_qwen3_reranker_;
};

SamplingParameters build_sampling_parameters(
    const std::vector<RequestSamplingParam>& request_params) {
  std::vector<const RequestSamplingParam*> request_param_ptrs;
  std::vector<int32_t> selected_token_idxes;
  std::vector<int32_t> sample_idxes;
  request_param_ptrs.reserve(request_params.size());
  selected_token_idxes.reserve(request_params.size());
  sample_idxes.reserve(request_params.size());

  for (size_t i = 0; i < request_params.size(); ++i) {
    request_param_ptrs.push_back(&request_params[i]);
    selected_token_idxes.push_back(static_cast<int32_t>(i));
    sample_idxes.push_back(static_cast<int32_t>(i));
  }

  const std::vector<std::vector<int64_t>> empty_unique_token_ids;
  const std::vector<std::vector<int32_t>> empty_unique_token_counts;
  const std::vector<int32_t> empty_unique_token_lens;

  SamplingParameters params;
  params.init(request_param_ptrs,
              selected_token_idxes,
              sample_idxes,
              empty_unique_token_ids,
              empty_unique_token_counts,
              empty_unique_token_lens);
  return params;
}

}  // namespace

TEST(SamplerTest, ResolveCandidateTokenIdsDeduplicatesAndKeepsOrder) {
  CandidateFlagsGuard guard;
  FLAGS_candidate_token_ids = "5,7,5,9";
  FLAGS_enable_qwen3_reranker = false;

  auto resolved = resolve_candidate_token_ids(/*vocab_size=*/64);

  EXPECT_EQ(resolved, (std::vector<int64_t>{5, 7, 9}));
}

TEST(SamplerTest, ResolveCandidateTokenIdsReturnsEmptyWhenUnset) {
  CandidateFlagsGuard guard;
  FLAGS_candidate_token_ids = "";
  FLAGS_enable_qwen3_reranker = false;

  auto resolved = resolve_candidate_token_ids(/*vocab_size=*/64);
  EXPECT_TRUE(resolved.empty());
}

TEST(SamplerTest, Qwen3RerankerUsesDefaultCandidateTokenIdsWhenUnset) {
  CandidateFlagsGuard guard;
  FLAGS_candidate_token_ids = "";
  FLAGS_enable_qwen3_reranker = true;

  auto resolved = resolve_candidate_token_ids(/*vocab_size=*/10000);

  EXPECT_EQ(FLAGS_candidate_token_ids, "2152,9693");
  EXPECT_EQ(resolved, (std::vector<int64_t>{2152, 9693}));
}

TEST(SamplerTest, ResolveCandidateTokenIdsRejectsOutOfRangeTokenId) {
  CandidateFlagsGuard guard;
  FLAGS_candidate_token_ids = "1,128";
  FLAGS_enable_qwen3_reranker = false;

  EXPECT_DEATH((void)resolve_candidate_token_ids(/*vocab_size=*/128),
               "out of vocab range");
}

TEST(SamplerTest, MapsNextTokensAndTopTokensToGlobalIds) {
  Sampler sampler;
  sampler.set_candidate_token_ids({10, 20, 30});

  RequestSamplingParam request_param;
  request_param.logprobs = true;
  request_param.top_logprobs = 2;
  auto params = build_sampling_parameters({request_param});

  auto logits = torch::tensor({{0.1f, 0.9f, 0.2f}}, torch::kFloat32);
  auto output = sampler.forward(logits, params);

  EXPECT_TRUE(
      torch::equal(output.next_tokens, torch::tensor({20}, torch::kLong)));
  ASSERT_TRUE(output.top_tokens.defined());
  EXPECT_TRUE(
      torch::equal(output.top_tokens, torch::tensor({{20, 30}}, torch::kLong)));
}

TEST(SamplerTest, Qwen3RerankerUsesCandidateIndexOneProbability) {
  Qwen3RerankerFlagGuard guard;
  FLAGS_enable_qwen3_reranker = true;

  Sampler sampler;
  sampler.set_candidate_token_ids({100, 200});

  RequestSamplingParam request_param;
  request_param.logprobs = true;
  auto params = build_sampling_parameters({request_param, request_param});
  auto logits = torch::tensor({{1.0f, 2.0f}, {3.0f, 1.0f}}, torch::kFloat32);
  auto output = sampler.forward(logits, params);

  auto probs = torch::softmax(logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  EXPECT_NEAR(
      output.logprobs[0].item<float>(), probs[0][1].item<float>(), 1e-6);
  EXPECT_NEAR(
      output.logprobs[1].item<float>(), probs[1][1].item<float>(), 1e-6);
  EXPECT_TRUE(torch::equal(output.next_tokens,
                           torch::tensor({200, 100}, torch::kLong)));
}

TEST(SamplerTest, Qwen3RerankerRequiresTwoCandidateLogits) {
  Qwen3RerankerFlagGuard guard;
  FLAGS_enable_qwen3_reranker = true;

  Sampler sampler;
  sampler.set_candidate_token_ids({10, 20, 30});

  RequestSamplingParam request_param;
  request_param.logprobs = true;
  auto params = build_sampling_parameters({request_param});

  auto logits = torch::tensor({{0.1f, 0.9f, 0.2f}}, torch::kFloat32);
  EXPECT_DEATH((void)sampler.forward(logits, params),
               "requires exactly two candidate logits");
}

}  // namespace xllm
