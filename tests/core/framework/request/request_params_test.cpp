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

#include "core/framework/request/request_params.h"

#include <gtest/gtest.h>

#include "chat.pb.h"
#include "completion.pb.h"

namespace xllm {
namespace {

TEST(RequestParamsTest,
     CompletionBeamSearchDefaultsTopLogprobsToBeamWidthWhenUnset) {
  proto::CompletionRequest request;
  request.set_beam_width(3);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 3);
}

TEST(RequestParamsTest, CompletionBeamSearchKeepsExplicitLogprobsWhenSet) {
  proto::CompletionRequest request;
  request.set_beam_width(3);
  request.set_logprobs(5);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 5);
}

TEST(RequestParamsTest, CompletionNonBeamSearchKeepsLogprobsDisabled) {
  proto::CompletionRequest request;
  request.set_beam_width(1);

  RequestParams params(request, "", "");

  EXPECT_FALSE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

TEST(RequestParamsTest, ChatBeamSearchDefaultsTopLogprobsToBeamWidthWhenUnset) {
  proto::ChatRequest request;
  request.set_beam_width(4);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 4);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitLogprobsDisabled) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(false);

  RequestParams params(request, "", "");

  EXPECT_FALSE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitTopLogprobs) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(true);
  request.set_top_logprobs(2);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 2);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitZeroTopLogprobs) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(true);
  request.set_top_logprobs(0);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

}  // namespace
}  // namespace xllm
