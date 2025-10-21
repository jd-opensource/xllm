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

#include <torch/torch.h>

#include <string>

#include "ATen/Tensor.h"
#include "torch_mlu_ops.h"
namespace xllm::mlu {

static const std::string kActModeSilu = "silu";
static const std::string kActModeGelu = "gelu";
static const std::string kActModeQuickGelu = "quick_gelu";
static const std::string kActModeSwish = "swish";

at::Tensor matmul(const at::Tensor& a,
                  const at::Tensor& b,
                  const std::optional<at::Tensor>& bias,
                  const std::optional<at::Tensor>& c,
                  double alpha,
                  double beta);

torch::Tensor fused_moe(
    const torch::Tensor& hidden_states,
    const torch::Tensor& gating_output,
    const torch::Tensor& w1,
    const torch::Tensor& w2,
    const std::optional<torch::Tensor>& bias1,
    const std::optional<torch::Tensor>& bias2,
    const std::optional<torch::Tensor>& residual,
    const std::optional<torch::Tensor>& input_smooth,
    const std::optional<torch::Tensor>& act_smooth,
    const std::optional<torch::Tensor>& w1_scale,
    const std::optional<torch::Tensor>& w2_scale,
    const std::optional<torch::Tensor>& e_score_correction_bias,
    int topk,
    bool renormalize,
    bool gated,
    const std::string& act_mode,
    const std::string& scoring_func = "softmax",
    int start_expert_id = 0,
    int block_n = 0,
    bool avg_moe = false,
    const std::optional<torch::Tensor>& class_reduce_weight = std::nullopt,
    const std::optional<torch::Tensor>& class_expert_id = std::nullopt,
    const std::optional<std::vector<bool>>& w1_quant_flag = std::nullopt,
    const std::optional<std::vector<bool>>& w2_quant_flag = std::nullopt,
    int world_size = 0,
    int shared_expert_num = 0,
    const std::string& parallel_mode = "ep");

}  // namespace xllm::mlu
