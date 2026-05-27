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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>

namespace xllm::runtime::detail {

inline void write_first_round_beam_outputs(
    const torch::Tensor& flat_top_tokens,
    const torch::Tensor& flat_top_logprobs,
    int32_t batch_size,
    torch::Tensor& out_token_ids,
    torch::Tensor& out_log_probs,
    torch::Tensor& out_seqgroup) {
  CHECK_EQ(flat_top_tokens.dim(), 2)
      << "flat_top_tokens must be [batch * beam, 1], got "
      << flat_top_tokens.sizes();
  CHECK_EQ(flat_top_logprobs.dim(), 2)
      << "flat_top_logprobs must be [batch * beam, 1], got "
      << flat_top_logprobs.sizes();
  CHECK_EQ(flat_top_tokens.sizes(), flat_top_logprobs.sizes())
      << "flat_top_tokens/top_logprobs shape mismatch";
  CHECK_EQ(flat_top_tokens.size(0), out_token_ids.size(0))
      << "top_tokens/out_token_ids rows mismatch";
  CHECK_EQ(flat_top_logprobs.size(0), out_log_probs.size(0))
      << "top_logprobs/out_log_probs rows mismatch";
  CHECK_EQ(flat_top_tokens.size(1), 1)
      << "first-round top_tokens must have exactly one column";
  CHECK_EQ(flat_top_logprobs.size(1), 1)
      << "first-round top_logprobs must have exactly one column";
  CHECK_GT(batch_size, 0) << "batch_size must be positive";
  CHECK_EQ(flat_top_tokens.size(0) % batch_size, 0)
      << "top_tokens rows must be divisible by batch_size";
  CHECK_EQ(out_seqgroup.dim(), 3)
      << "out_seqgroup must be [batch, beam, rounds], got "
      << out_seqgroup.sizes();
  CHECK_EQ(out_seqgroup.size(0), batch_size) << "out_seqgroup batch mismatch";

  const int64_t beam_width = flat_top_tokens.size(0) / batch_size;
  CHECK_EQ(out_seqgroup.size(1), beam_width) << "out_seqgroup beam mismatch";

  out_token_ids.copy_(flat_top_tokens);
  out_log_probs.copy_(flat_top_logprobs);
  out_seqgroup.select(/*dim=*/2, /*index=*/0)
      .copy_(flat_top_tokens.view({batch_size, beam_width}));
}

}  // namespace xllm::runtime::detail
