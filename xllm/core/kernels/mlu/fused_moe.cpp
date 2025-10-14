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

#include "torch_mlu/csrc/aten/cnnl/cnnlHandle.h"
#include "torch_mlu/csrc/framework/core/mlu_guard.h"
#include "torch_mlu_ops.h"
#include "torch_ops_api.h"

namespace {
torch::Tensor create_group_gemm_output(const torch::Tensor& a,
                                       const torch::Tensor& b,
                                       const torch::Tensor& group_list) {
  if (b.dim() != 2) {
    return torch::empty({a.size(0), b.size(1)}, a.options());
  }
  return torch::empty({group_list.size(0), a.size(0), b.size(0)}, a.options());
}
}  // namespace

namespace xllm::mlu {
torch::Tensor fused_moe(const torch::Tensor& hidden_states,
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
                        int topk,
                        bool renormalize,
                        bool gated,
                        const std::string& act_mode,
                        const std::string& scoring_func,
                        int start_expert_id,
                        int block_n,
                        bool avg_moe,
                        const std::optional<torch::Tensor>& class_reduce_weight,
                        const std::optional<torch::Tensor>& class_expert_id,
                        const std::optional<std::vector<bool>>& w1_quant_flag,
                        const std::optional<std::vector<bool>>& w2_quant_flag,
                        int world_size,
                        int shared_expert_num,
                        const std::string& parallel_mode) {
  auto dtype = hidden_states.dtype();
  auto ori_input_shape = hidden_states.sizes();

  auto hidden_states_2d = hidden_states.reshape({-1, hidden_states.size(-1)});
  int64_t tokens = hidden_states_2d.size(0);
  auto gating_output_2d = gating_output.reshape({-1, gating_output.size(-1)});

  std::optional<torch::Tensor> residual_2d = std::nullopt;
  if (residual.has_value()) {
    residual_2d = residual.value().reshape({-1, residual.value().size(-1)});
  }

  int64_t expert_num = gating_output_2d.size(-1);
  int64_t expert_size = w1.size(0);

  // softmax_topk
  auto reduce_weight = torch::empty(
      {gating_output_2d.size(0), topk},
      torch::dtype(torch::kFloat).device(gating_output_2d.device()));
  auto expert_id = torch::empty(
      {gating_output_2d.size(0), topk},
      torch::dtype(torch::kInt32).device(gating_output_2d.device()));
  tmo::torch_api::moe_active_topk(gating_output_2d,
                                  topk,
                                  -1,
                                  0,
                                  renormalize,
                                  std::nullopt,
                                  "topk_logit",
                                  scoring_func,
                                  1.0,
                                  std::nullopt,
                                  reduce_weight,
                                  expert_id);

  auto output_vec = tmo::torch_api::moe_gen_idx(expert_id, expert_num);
  auto expand_idx = output_vec[0];
  auto combine_idx = output_vec[1];
  auto token_count = output_vec[2];
  auto cusum_token_count = output_vec[3];
  torch::Tensor expand_hidden_states =
      tmo::torch_api::moe_expand_input(hidden_states_2d,
                                       expand_idx,
                                       cusum_token_count,
                                       start_expert_id,
                                       expert_size);

  auto token_count_slice =
      token_count.slice(0, start_expert_id, start_expert_id + expert_size);
  torch::Tensor gemm1_out =
      create_group_gemm_output(expand_hidden_states, w1, token_count_slice);
  tmo::torch_api::group_gemm(expand_hidden_states,
                             w1,
                             token_count_slice,
                             gemm1_out,
                             std::nullopt,  // expand_idx: None
                             std::nullopt,  // c: None
                             std::nullopt,  // alpha: None
                             std::nullopt,  // beta: None
                             c10::nullopt,
                             c10::nullopt,
                             c10::nullopt,
                             c10::nullopt,
                             c10::nullopt,
                             c10::nullopt,
                             tokens,
                             false,
                             true);

  torch::Tensor act_out;
  if (gated) {
    act_out = gemm1_out.slice(1, 0, gemm1_out.size(1) / 2);
  } else {
    act_out = gemm1_out;
  }
  tmo::torch_api::active(gemm1_out,
                         act_out,
                         bias1,
                         cusum_token_count,
                         act_mode,
                         gated,
                         start_expert_id,
                         expert_size);

  torch::Tensor gemm2_out =
      create_group_gemm_output(act_out, w2, token_count_slice);
  tmo::torch_api::group_gemm(act_out,
                             w2,
                             token_count_slice,
                             gemm2_out,     // d
                             std::nullopt,  // expand_idx: None
                             std::nullopt,  // c: None
                             std::nullopt,  // alpha: None
                             std::nullopt,  // beta: None
                             c10::nullopt,
                             c10::nullopt,
                             c10::nullopt,
                             c10::nullopt,
                             c10::nullopt,
                             c10::nullopt,
                             tokens,
                             false,
                             true);

  auto output = torch::empty({reduce_weight.size(0), gemm2_out.size(1)},
                             gemm2_out.options());
  tmo::torch_api::moe_combine_result(gemm2_out,
                                     output,
                                     reduce_weight,
                                     combine_idx,
                                     residual_2d,
                                     cusum_token_count,
                                     start_expert_id,
                                     expert_size,
                                     bias2);

  return output.reshape(ori_input_shape);
}

}  // namespace xllm::mlu
