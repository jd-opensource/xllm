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

#include <glog/logging.h>

#include "mlu_ops_api.h"

namespace {
torch::Tensor create_group_gemm_output(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& group_list,
    torch::ScalarType dtype = torch::ScalarType::BFloat16) {
  torch::TensorOptions target_options = a.options().dtype(dtype);
  if (b.dim() != 2) {
    return torch::empty({a.size(0), b.size(1)}, target_options);
  }
  return torch::empty({group_list.size(0), a.size(0), b.size(0)},
                      target_options);
}
}  // namespace

namespace xllm::kernel::mlu {
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
    int64_t topk,
    bool renormalize,
    bool gated,
    const std::string& act_mode,
    const std::string& scoring_func,
    int64_t num_expert_group,
    int64_t topk_group,
    double route_scale,
    int64_t start_expert_id,
    bool avg_moe,
    const std::optional<torch::List<int64_t>>& w1_quant_flag,
    const std::optional<torch::List<int64_t>>& w2_quant_flag) {
  auto dtype = hidden_states.dtype();
  auto ori_input_shape = hidden_states.sizes();

  auto hidden_states_2d = hidden_states.reshape({-1, hidden_states.size(-1)});
  int64_t tokens = hidden_states_2d.size(0);
  auto gating_output_2d = gating_output.reshape({-1, gating_output.size(-1)});

  std::optional<torch::Tensor> residual_2d = std::nullopt;
  if (residual.has_value()) {
    residual_2d = residual.value().reshape({-1, residual.value().size(-1)});
  }

  // check smooth quant variables
  bool all_present = input_smooth && act_smooth && w1_scale && w2_scale;
  bool all_none = !input_smooth && !act_smooth && !w1_scale && !w2_scale;
  CHECK(all_none || all_present)
      << "input_smooth, act_smooth, w1_scale and w2_scale must be present or "
         "absent at the same time.";
  bool is_smoothquant = all_present;
  int64_t expert_num = gating_output_2d.size(-1);
  int64_t expert_size = w1.size(0);

  // apply softmax_topk or sigmoid_topk
  auto reduce_weight = torch::empty(
      {gating_output_2d.size(0), topk},
      torch::dtype(torch::kFloat).device(gating_output_2d.device()));
  auto expert_id = torch::empty(
      {gating_output_2d.size(0), topk},
      torch::dtype(torch::kInt32).device(gating_output_2d.device()));

  tmo::torch_api::moe_active_topk(gating_output_2d,
                                  topk,
                                  num_expert_group,
                                  topk_group,
                                  renormalize,
                                  /*mask=*/std::nullopt,
                                  /*normed_by=*/"topk_logit",
                                  scoring_func,
                                  route_scale,
                                  e_score_correction_bias,
                                  reduce_weight,
                                  expert_id);

  auto output_vec = tmo::torch_api::moe_gen_idx(expert_id, expert_num);
  auto expand_idx = output_vec[0];
  auto combine_idx = output_vec[1];
  auto token_count = output_vec[2];
  auto cusum_token_count = output_vec[3];

  // prepare the parameters for the first group gemm
  auto token_count_slice =
      token_count.slice(0, start_expert_id, start_expert_id + expert_size);
  auto gather_index_start_position =
      cusum_token_count.index({start_expert_id}).unsqueeze(0);
  torch::Tensor expand_hidden_states;
  torch::Tensor input_scale;

  if (is_smoothquant) {
    // w8a8 path: quantize input hidden states directly (fused with
    // moe_expand_input)
    std::tie(expand_hidden_states, input_scale) =
        scaled_quantize(hidden_states_2d,  // Use original hidden_states_2d
                                           // instead of expand_hidden_states
                        input_smooth.value(),
                        /*zero=*/std::nullopt,
                        token_count_slice,
                        expand_idx,
                        gather_index_start_position,
                        /*output=*/std::nullopt,
                        /*output_scale=*/std::nullopt,
                        /*act_mode=*/"none",
                        /*active_coef=*/1.0,
                        /*is_gated=*/false,
                        /*quant_type=*/torch::kChar);
  } else {
    // bf16/fp32 path: expand input hidden states
    expand_hidden_states = tmo::torch_api::moe_expand_input(hidden_states_2d,
                                                            expand_idx,
                                                            cusum_token_count,
                                                            start_expert_id,
                                                            expert_size);
  }

  torch::Tensor gemm1_out = create_group_gemm_output(
      expand_hidden_states, w1, token_count_slice, dtype.toScalarType());

  // Unified group_gemm call using input_scale/w1_scale/quant_flag only if
  // present
  tmo::torch_api::group_gemm(
      expand_hidden_states,
      w1,
      token_count_slice,
      gemm1_out,
      /*gather_idx=*/std::nullopt,
      /*c=*/std::nullopt,
      /*alpha=*/std::nullopt,
      /*beta=*/std::nullopt,
      /*a_scale=*/input_scale.defined() ? std::make_optional(input_scale)
                                        : std::nullopt,
      /*b_scale=*/w1_scale.has_value() ? std::make_optional(w1_scale.value())
                                       : std::nullopt,
      /*bias=*/std::nullopt,
      /*a_calibration=*/std::nullopt,
      /*b_calibration=*/std::nullopt,
      /*quant_flag=*/w1_quant_flag.has_value() ? w1_quant_flag : std::nullopt,
      /*b_offset=*/std::nullopt,
      /*tile_config=*/std::nullopt,
      /*max_dim=*/tokens,
      /*trans_a=*/false,
      /*trans_b=*/true,
      /*a_quant_bit=*/is_smoothquant ? 8 : -1);

  // prepare the parameters for the second group gemm
  torch::Tensor act_out;
  torch::Tensor act_out_scale;
  if (is_smoothquant) {
    // w8a8 path: reuse quantized_input and input_scale from first group_gemm
    act_out = gated ? expand_hidden_states.slice(1, 0, gemm1_out.size(1) / 2)
                    : expand_hidden_states.slice(1, 0, gemm1_out.size(1));
    act_out_scale = input_scale.slice(0, 0, gemm1_out.size(0));

    // Quantize gemm1_out directly (fused with active operation) using reused
    // tensors
    auto [quantized_activation, activation_scale] =
        scaled_quantize(gemm1_out,
                        act_smooth.value(),
                        /*zero=*/std::nullopt,
                        /*token_count=*/token_count_slice,
                        /*gather_index=*/std::nullopt,
                        /*gather_index_start_position=*/std::nullopt,
                        act_out,        // output - reuse from quantized_input
                        act_out_scale,  // output_scale - reuse from input_scale
                        /*act_mode=*/act_mode,
                        /*active_coef=*/1.0,
                        /*is_gated=*/gated,
                        /*quant_type=*/torch::kChar);
    act_out = quantized_activation;
    act_out_scale = activation_scale;
  } else {
    // bf16/fp32 path: apply activation function first
    act_out = gated ? gemm1_out.slice(1, 0, gemm1_out.size(1) / 2) : gemm1_out;
    tmo::torch_api::active(gemm1_out,
                           act_out,
                           bias1,
                           cusum_token_count,
                           act_mode,
                           gated,
                           start_expert_id,
                           expert_size);
  }

  torch::Tensor gemm2_out = create_group_gemm_output(
      act_out, w2, token_count_slice, dtype.toScalarType());

  // Unified group_gemm call, now only checks the existance of
  // input_scale/w1_scale for smoothquant
  tmo::torch_api::group_gemm(
      act_out,
      w2,
      token_count_slice,
      gemm2_out,
      /*gather_idx=*/std::nullopt,
      /*c=*/std::nullopt,
      /*alpha=*/std::nullopt,
      /*beta=*/std::nullopt,
      act_out_scale.defined() ? std::make_optional(act_out_scale)
                              : std::nullopt,  // a_scale
      w2_scale.has_value() ? std::make_optional(w2_scale.value())
                           : std::nullopt,  // b_scale
      /*bias=*/std::nullopt,
      /*a_calibration=*/std::nullopt,
      /*b_calibration=*/std::nullopt,
      w2_quant_flag.has_value() ? w2_quant_flag : std::nullopt,  // quant_flag
      /*b_offset=*/std::nullopt,
      /*tile_config=*/std::nullopt,
      /*max_dim=*/tokens,
      /*trans_a=*/false,
      /*trans_b=*/true,
      /*a_quant_bit=*/is_smoothquant ? 8 : -1);

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

}  // namespace xllm::kernel::mlu
