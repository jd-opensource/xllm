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

#include "fused_moe.h"

#include <glog/logging.h>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(int64_t num_experts,
                           int64_t top_k,
                           int64_t num_expert_group,
                           int64_t topk_group,
                           double route_scale,
                           int64_t hidden_size,
                           int64_t intermediate_size,
                           int64_t n_shared_experts,
                           bool is_gated,
                           bool has_score_bias,
                           bool has_bias,
                           bool skip_bias_add,
                           int64_t renormalize,
                           const std::string& hidden_act,
                           const std::string& scoring_func,
                           const std::string& topk_method,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : topk_(top_k),
      num_expert_group_(num_expert_group),
      topk_group_(topk_group),
      route_scale_(route_scale),
      n_shared_experts_(n_shared_experts),
      is_gated_(is_gated),
      has_score_bias_(has_score_bias),
      has_bias_(has_bias),
      skip_bias_add_(skip_bias_add),
      renormalize_(renormalize),
      hidden_act_(hidden_act),
      scoring_func_(scoring_func),
      quant_args_(quant_args),
      parallel_args_(parallel_args),
      options_(options) {
  int64_t ep_size = parallel_args.ep_size();
  int64_t ep_rank = 0;
  tp_pg_ = parallel_args.tp_group_;
  if (ep_size > 1) {
    ep_rank = parallel_args.moe_ep_group_->rank();
    tp_pg_ = parallel_args.moe_tp_group_;
  }

  // smoothquant check: If quant_method is not empty, only w8a8 smoothquant is
  // supported
  if (!quant_args.quant_method().empty()) {
    if (quant_args.quant_method() != "smoothquant" || quant_args.bits() != 8 ||
        !quant_args.activation_dynamic()) {
      LOG(FATAL) << "FusedMoE only supports w8a8 smoothquant quantization when "
                    "quant_method is set. "
                 << "Got quant_method=" << quant_args.quant_method()
                 << ", bits=" << quant_args.bits()
                 << ", activation_dynamic=" << quant_args.activation_dynamic();
    }
    // If confirmed as smoothquant w8a8, set is_smoothquant_ to true
    is_smoothquant_ = true;
  } else {
    is_smoothquant_ = false;
  }

  // calculate the number of experts per rank
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ = register_parameter(
        "e_score_correction_bias", torch::empty({num_experts}, options), false);
  }

  gate_ = register_module(
      "gate_proj",
      ReplicatedLinear(hidden_size, num_experts, false, quant_args, options));
  if (n_shared_experts_ > 0) {
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size,
                                 intermediate_size * n_shared_experts_,
                                 is_gated_,
                                 false,
                                 hidden_act_,
                                 quant_args,
                                 parallel_args,
                                 options));
  }

  // create weight buffer
  const int64_t world_size = tp_pg_->world_size();
  int64_t local_intermediate_size = intermediate_size / world_size;
  if (is_smoothquant_) {
    auto quant_option = options_.dtype(torch::kInt8);
    auto fp_option = options_.dtype(torch::kFloat32);
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size},
            quant_option),
        false);
    w13_scale_ = register_parameter(
        "w13_scale",
        torch::empty({num_experts_per_rank_, local_intermediate_size * 2},
                     fp_option),
        false);
    input_smooth_ = register_parameter(
        "input_smooth",
        torch::empty({num_experts_per_rank_, hidden_size}, fp_option),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size, local_intermediate_size},
            quant_option),
        false);
    w2_scale_ = register_parameter(
        "w2_scale",
        torch::empty({num_experts_per_rank_, hidden_size}, fp_option),
        false);
    act_smooth_ = register_parameter(
        "act_smooth",
        torch::empty({num_experts_per_rank_, local_intermediate_size},
                     fp_option),
        false);

  } else {
    w13_ = register_parameter(
        "w13",
        torch::empty(
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size},
            options_),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size, local_intermediate_size},
            options_),
        false);
  }
}

torch::Tensor FusedMoEImpl::forward_expert(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& shared_output) {
  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }

  xllm::kernel::FusedMoEParams fused_moe_params;
  fused_moe_params.hidden_states = hidden_states;
  fused_moe_params.gating_output = router_logits;
  fused_moe_params.w1 = w13_;
  fused_moe_params.w2 = w2_;
  fused_moe_params.residual = shared_output;
  fused_moe_params.num_expert_group = num_expert_group_;
  fused_moe_params.topk_group = topk_group_;
  fused_moe_params.route_scale = route_scale_;
  fused_moe_params.e_score_correction_bias = e_score_correction_bias;
  fused_moe_params.topk = topk_;
  fused_moe_params.renormalize = renormalize_;
  fused_moe_params.gated = is_gated_;
  fused_moe_params.act_mode = hidden_act_;
  fused_moe_params.scoring_func = scoring_func_;
  fused_moe_params.start_expert_id = start_expert_id_;
  if (is_smoothquant_) {
    fused_moe_params.w1_scale = w13_scale_;
    fused_moe_params.w2_scale = w2_scale_;
    fused_moe_params.input_smooth = input_smooth_;
    fused_moe_params.act_smooth = act_smooth_;
  }

  auto final_hidden_states = xllm::kernel::fused_moe(fused_moe_params);

  if (tp_pg_->world_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
  }
  if (parallel_args_.ep_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states,
                                                 parallel_args_.moe_ep_group_);
  }
  return final_hidden_states;
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& input_params) {
  auto input = hidden_states;
  bool need_slice = false;
  if (parallel_args_.dp_size() > 1 && parallel_args_.ep_size() > 1) {
    input = parallel_state::gather(input,
                                   parallel_args_.dp_local_process_group_,
                                   input_params.dp_global_token_nums);
    need_slice = true;
  }

  std::optional<torch::Tensor> shared_output = std::nullopt;
  if (n_shared_experts_ > 0) {
    shared_output = shared_experts_(input);
  }
  auto router_logits = gate_(input);
  auto output = forward_expert(input, router_logits, shared_output);

  if (need_slice) {
    const auto& dp_tokens = input_params.dp_global_token_nums;
    const int64_t dp_rank = parallel_args_.dp_local_process_group_->rank();
    auto start =
        std::accumulate(dp_tokens.begin(), dp_tokens.begin() + dp_rank, 0);
    auto end = start + dp_tokens[dp_rank];
    output = output.slice(0, start, end);
  }
  return output;
}

void FusedMoEImpl::load_e_score_correction_bias(const StateDict& state_dict) {
  if (e_score_correction_bias_.defined() &&
      !e_score_correction_bias_is_loaded_) {
    LOAD_WEIGHT(e_score_correction_bias);
  }
}

void FusedMoEImpl::load_experts(const StateDict& state_dict) {
  const int64_t rank = tp_pg_->rank();
  const int64_t world_size = tp_pg_->world_size();
  const int64_t start_expert_id = start_expert_id_;
  const int64_t num_experts_per_rank = num_experts_per_rank_;
  std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  if (is_smoothquant_) {
    LOAD_MOE_FUSED_WEIGHT("qweight", w1, w3, w13);
    LOAD_MOE_FUSED_WEIGHT("per_channel_scale", w1_scale, w3_scale, w13_scale);
    LOAD_MOE_WEIGHT("up_proj.", "smooth", input_smooth, -1);
    LOAD_MOE_WEIGHT("down_proj.", "qweight", w2, 1);
    LOAD_MOE_WEIGHT("down_proj.", "per_channel_scale", w2_scale, -1);
    LOAD_MOE_WEIGHT("down_proj.", "smooth", act_smooth, 0);
  } else {
    LOAD_MOE_FUSED_WEIGHT("weight", w1, w3, w13);
    LOAD_MOE_WEIGHT("down_proj.", "weight", w2, 1);
  }
}

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  if (n_shared_experts_ > 0) {
    shared_experts_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_experts."));
  }
  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_e_score_correction_bias(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
