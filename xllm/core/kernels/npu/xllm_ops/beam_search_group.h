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
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <vector>

#include "acl/acl.h"
#include "aclnn_beam_search_group.h"
#include "acltensor_utils.h"
#include "util/tensor_helper.h"

namespace xllm_ops {
void beam_search(const torch::Tensor& logprobs,
                 const torch::Tensor& top_tokens,
                 const torch::Tensor& top_logprobs,
                 torch::Tensor& sequence_group,
                 int64_t current_step,
                 torch::Tensor& out_token_ids,
                 torch::Tensor& out_token_index,
                 torch::Tensor& out_log_probs,
                 torch::Tensor& out_beam_count_prefix_sums,
                 torch::Tensor& out_sequence);
}  // namespace xllm_ops