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

#include "dp_utils.h"

#include <numeric>

#include "core/util/utils.h"
#include "framework/parallel_state/parallel_state.h"

namespace xllm {
namespace layer {

std::pair<torch::Tensor, PaddingInfo> check_and_pad_before_scatter(
    torch::Tensor x,
    const ParallelArgs& parallel_args) {
  int64_t tp_size = parallel_args.tp_group_->world_size();
  int64_t current_tokens = x.size(0);

  // Calculate the target token count to be aligned to tp_size.
  // Logic: (current_tokens + tp_size - 1) / tp_size * tp_size
  // This rounds up current_tokens to the nearest multiple of tp_size.
  // Examples: Token=2, TP=4 -> target=4; Token=5, TP=4 -> target=8
  int64_t target_tokens = (current_tokens + tp_size - 1) / tp_size * tp_size;

  // If already aligned (or zero), no need to pad.
  // (Depending on the operator, zero may or may not be acceptable, but we
  // assume at least some data.)
  if (target_tokens == current_tokens) {
    return {x, {current_tokens, current_tokens, false}};
  }

  int64_t pad_len = target_tokens - current_tokens;

  // Construct the padding tensor filled with zeros to avoid affecting Reduce
  // phase summation results. Assuming x is of float/half type, so generate a
  // [pad_len, hidden_dim] zero tensor. IMPORTANT: Must use zero padding because
  // Reduce-Scatter does sum operations and zero won't contaminate real data.
  torch::Tensor pad_tensor = torch::zeros({pad_len, x.size(1)}, x.options());

  torch::Tensor padded_x = torch::cat({x, pad_tensor}, 0);

  return {padded_x, {current_tokens, target_tokens, true}};
}

torch::Tensor check_and_unpad_after_gather(torch::Tensor x,
                                           const PaddingInfo& pad_info) {
  if (!pad_info.active) {
    return x;
  }

  // slice out [0, original_tokens]
  return x.slice(0, 0, pad_info.original_tokens);
}

torch::Tensor get_dp_local_slice(const torch::Tensor& input,
                                 const ModelInputParams& params,
                                 const ParallelArgs& args) {
  // If data parallelism is not enabled, return input as is (no slicing needed)
  if (args.dp_size() <= 1) {
    return input;
  }

  // Get the global token count for each DP rank and the current rank id
  const auto& dp_tokens = params.dp_global_token_nums;
  const int64_t dp_rank = args.dp_local_process_group_->rank();

  // Compute the range for this rank: start offset = sum of previous ranks'
  // tokens
  int64_t start = 0;
  for (int i = 0; i < dp_rank; ++i) {
    start += dp_tokens[i];
  }
  int64_t end = start + dp_tokens[dp_rank];

  // Slice and return a view of the input corresponding to this local DP rank
  return input.slice(0, start, end);
}

}  // namespace layer
}  // namespace xllm
