/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

#include "core/common/types.h"

namespace xllm {
struct Usage {
  // the number of tokens in the prompt.
  size_t num_prompt_tokens = 0;

  // the number of tokens in the generated completion.
  size_t num_generated_tokens = 0;

  // the total number of tokens used in the request (prompt + completion).
  size_t num_total_tokens = 0;
};

struct LogProbData {
  // the text of the token.
  std::string token;
  // the token id.
  int32_t token_id;
  // the log probability of the token.
  float logprob = -9999.0f;
  // whether the token is finished.
  bool finished_token = true;
};

struct LogProb : public LogProbData {
  // the top log probabilities.
  std::optional<std::vector<LogProbData>> top_logprobs;
};

struct SequenceOutput {
  // the index of the sequence in the request.
  size_t index;

  // the generated/delta text.
  // delta text is the text generated since the last response for streaming.
  std::string text;

  torch::Tensor embedding;

  // the token ids of the generated text.
  std::vector<int32_t> token_ids;

  // item_id for rec.
  std::optional<int64_t> item_ids;

  // the reason the sequence finished.
  std::optional<std::string> finish_reason;

  // log probabilities of the generated tokens.
  std::optional<std::vector<LogProb>> logprobs;

  // the embeddings of the prompt token
  std::optional<std::vector<float>> embeddings;
};

struct RequestOutput {
  RequestOutput() = default;

  RequestOutput(Status&& _status) : status(std::move(_status)) {}

  void log_request_status() const;

  // the id of the request.
  std::string request_id;

  // the id of the request which generated in xllm service.
  std::string service_request_id;

  // the prompt text for the request.
  std::optional<std::string> prompt;

  // the status of the request.
  std::optional<Status> status;

  // the output for each sequence in the request.
  std::vector<SequenceOutput> outputs;

  // the statistics for the request.
  std::optional<Usage> usage;

  // whether the request is finished.
  bool finished = false;

  // whether the request is cancelled.
  bool cancelled = false;
};

// callback function for output, return true to continue, false to stop/cancel
using OutputCallback = std::function<bool(RequestOutput output)>;

using BatchOutputCallback =
    std::function<bool(size_t index, RequestOutput output)>;

}  // namespace xllm
