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

#pragma once
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"

namespace xllm {
namespace layer {

// Used to record information before padding
struct PaddingInfo {
  int64_t original_tokens = 0;  // Number of tokens before padding
  int64_t padded_tokens = 0;    // Number of tokens after padding
  bool active = false;          // Whether padding was performed
};

// Padding logic before Reduce Scatter
// Ensure that the number of tokens is a multiple of the TP group size,
// and at least equal to the TP size (so that each rank gets at least one token)
std::pair<torch::Tensor, PaddingInfo> check_and_pad_before_scatter(
    torch::Tensor x,
    const ParallelArgs& parallel_args);

// Unpadding logic after All Gather
// Simply slice out the original length
torch::Tensor check_and_unpad_after_gather(torch::Tensor x,
                                           const PaddingInfo& pad_info);

// given a tensor containing data from all DP ranks,
// returns a slice containing only the tokens for the current DP rank
torch::Tensor get_dp_local_slice(const torch::Tensor& input,
                                 const ModelInputParams& params,
                                 const ParallelArgs& args);

}  // namespace layer
}  // namespace xllm
