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

FusedMoEImpl::FusedMoEImpl(int num_experts,
                           int top_k,
                           int num_expert_group,
                           int topk_group,
                           double route_scale,
                           int hidden_size,
                           int intermediate_size,
                           int n_shared_experts,
                           bool is_gated,
                           bool has_score_bias,
                           bool has_bias,
                           bool skip_bias_add,
                           int renormalize,
                           const std::string& hidden_act,
                           const std::string& scoring_func,
                           const std::string& topk_method,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_experts_(num_experts),
      topk_(top_k),
      num_expert_group_(num_expert_group),
      topk_group_(topk_group),
      route_scale_(route_scale),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
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
  int ep_size = parallel_args.ep_size();
  int ep_rank = 0;
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
  num_experts_per_rank_ = num_experts_ / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  w13_list_.resize(num_experts_per_rank_);
  w1_list_.resize(num_experts_per_rank_);
  w3_list_.resize(num_experts_per_rank_);
  w2_list_.resize(num_experts_per_rank_);

  if (is_smoothquant_) {
    // Initialize scale and smooth lists for smoothquant
    w13_scale_list_.resize(num_experts_per_rank_);
    w1_scale_list_.resize(num_experts_per_rank_);
    w3_scale_list_.resize(num_experts_per_rank_);
    w2_scale_list_.resize(num_experts_per_rank_);
    input_smooth_list_.resize(num_experts_per_rank_);
    act_smooth_list_.resize(num_experts_per_rank_);
  }

  if (topk_method == "noaux_tc") {
    e_score_correction_bias_ =
        register_parameter("e_score_correction_bias",
                           torch::empty({num_experts_}, options),
                           /*requires_grad=*/false);
  }

  gate_ = register_module(
      "gate_proj",
      ReplicatedLinear(hidden_size_, num_experts_, false, quant_args, options));
  if (n_shared_experts_ > 0) {
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size_ * n_shared_experts_,
                                 is_gated_,
                                 false,
                                 hidden_act_,
                                 quant_args,
                                 parallel_args,
                                 options));
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
  pack_params();

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

  pack_params();
  std::optional<torch::Tensor> shared_output = std::nullopt;
  if (n_shared_experts_ > 0) {
    shared_output = shared_experts_(input);
  }
  auto router_logits = gate_(input);
  auto output = forward_expert(input, router_logits, shared_output);

  if (need_slice) {
    const auto& dp_tokens = input_params.dp_global_token_nums;
    const int dp_rank = parallel_args_.dp_local_process_group_->rank();
    auto start =
        std::accumulate(dp_tokens.begin(), dp_tokens.begin() + dp_rank, 0);
    auto end = start + dp_tokens[dp_rank];
    output = output.slice(0, start, end);
  }
  return output;
}

void FusedMoEImpl::pack_params() {
  if (w13_is_loaded_) {
    return;
  }
  w13_ = map_param_data(w13_list_);
  w2_ = map_param_data(w2_list_);
  if (is_smoothquant_) {
    w13_scale_ = map_param_data(w13_scale_list_);
    w2_scale_ = map_param_data(w2_scale_list_);
    input_smooth_ = map_param_data(input_smooth_list_);
    act_smooth_ = map_param_data(act_smooth_list_);
  }
  w13_is_loaded_ = true;

  w13_list_.clear();
  w2_list_.clear();
  w1_list_.clear();
  w3_list_.clear();
  w13_scale_list_.clear();
  w2_scale_list_.clear();
  input_smooth_list_.clear();
  act_smooth_list_.clear();
}

torch::Tensor FusedMoEImpl::map_param_data(
    const std::vector<torch::Tensor>& param_list) {
  if (param_list.size() == 1) {
    return param_list[0].unsqueeze(0);
  }

  std::vector<torch::Tensor> flattened_params;
  for (const auto& param : param_list) {
    CHECK(param.defined())
        << "Parameter(w13/w2) is not defined, please check the state dict.";
    flattened_params.push_back(param.data().view(-1));
  }
  auto packed_param = torch::cat(flattened_params);

  auto shape = param_list[0].sizes().vec();
  shape.insert(shape.begin(), param_list.size());
  return packed_param.view(shape);
}

void FusedMoEImpl::load_w13(const StateDict& state_dict,
                            int idx,
                            bool is_gated) {
  if (w13_list_[idx].defined() || state_dict.size() == 0) {
    return;
  }
  const auto rank = tp_pg_->rank();
  const auto world_size = tp_pg_->world_size();
  // load w1 and w3 weights
  if (is_smoothquant_) {
    DEFINE_WEIGHT(qweight);
    DEFINE_WEIGHT(per_channel_scale);
    qweight_ = torch::empty({intermediate_size_ / world_size, hidden_size_},
                            options_.dtype(torch::kInt8));
    per_channel_scale_ = torch::empty({intermediate_size_ / world_size},
                                      options_.dtype(torch::kFloat32));
    LOAD_SHARDED_WEIGHT(qweight, 0);
    LOAD_SHARDED_WEIGHT(per_channel_scale, 0);

    // load smooth value that does not need to be sharded
    DEFINE_WEIGHT(smooth);
    smooth_ = torch::empty({hidden_size_}, options_.dtype(torch::kFloat32));
    LOAD_WEIGHT(smooth);

    if (!qweight_is_loaded_ || !per_channel_scale_is_loaded_ ||
        !smooth_is_loaded_) {
      LOG(ERROR) << "qweight, per_channel_scale, or smooth was not loaded "
                    "successfully. Please check if all required weights are "
                    "present in the state dict.";
      return;
    }

    if (is_gated) {
      w1_list_[idx] = qweight_;
      w1_scale_list_[idx] = per_channel_scale_;
      // only need to load one smooth value,
      // because gate and up_proj shared the same input smooth
      input_smooth_list_[idx] = smooth_;
    } else {
      w3_list_[idx] = qweight_;
      w3_scale_list_[idx] = per_channel_scale_;
    }
  } else {
    DEFINE_WEIGHT(weight);
    weight_ =
        torch::empty({intermediate_size_ / world_size, hidden_size_}, options_);
    LOAD_SHARDED_WEIGHT(weight, 0);

    if (!weight_is_loaded_) {
      return;
    }

    if (is_gated) {
      w1_list_[idx] = weight_;
    } else {
      w3_list_[idx] = weight_;
    }
  }

  // combine w1 and w3 weights into w13
  if (w1_list_[idx].defined() && w3_list_[idx].defined()) {
    w13_list_[idx] = torch::cat({w1_list_[idx], w3_list_[idx]});
    w1_list_[idx].reset();
    w3_list_[idx].reset();
  }
  if (is_smoothquant_) {
    if (w1_scale_list_[idx].defined() && w3_scale_list_[idx].defined()) {
      w13_scale_list_[idx] =
          torch::cat({w1_scale_list_[idx], w3_scale_list_[idx]});
      w1_scale_list_[idx].reset();
      w3_scale_list_[idx].reset();
    }
  }
}

void FusedMoEImpl::load_w2(const StateDict& state_dict, int idx) {
  if (w2_list_[idx].defined() || state_dict.size() == 0) {
    return;
  }
  const auto rank = tp_pg_->rank();
  const auto world_size = tp_pg_->world_size();
  if (is_smoothquant_) {
    DEFINE_WEIGHT(qweight);
    DEFINE_WEIGHT(smooth);
    qweight_ = torch::empty({hidden_size_, intermediate_size_ / world_size},
                            options_.dtype(torch::kInt8));
    smooth_ = torch::empty({intermediate_size_ / world_size},
                           options_.dtype(torch::kFloat32));
    LOAD_SHARDED_WEIGHT(qweight, 1);
    LOAD_SHARDED_WEIGHT(smooth, 0);

    // load per_channel_scale value that does not need to be sharded
    DEFINE_WEIGHT(per_channel_scale);
    per_channel_scale_ =
        torch::empty({hidden_size_}, options_.dtype(torch::kFloat32));
    LOAD_WEIGHT(per_channel_scale);

    if (!qweight_is_loaded_ || !per_channel_scale_is_loaded_ ||
        !smooth_is_loaded_) {
      LOG(ERROR) << "qweight, per_channel_scale, or smooth was not loaded "
                    "successfully. Please check if all required weights are "
                    "present in the state dict.";
      return;
    }
    w2_list_[idx] = qweight_;
    w2_scale_list_[idx] = per_channel_scale_;
    act_smooth_list_[idx] = smooth_;
  } else {
    DEFINE_WEIGHT(weight);
    weight_ =
        torch::empty({hidden_size_, intermediate_size_ / world_size}, options_);
    LOAD_SHARDED_WEIGHT(weight, 1);
    if (!weight_is_loaded_) {
      return;
    }
    w2_list_[idx] = weight_;
  }
}

void FusedMoEImpl::load_e_score_correction_bias(const StateDict& state_dict) {
  if (e_score_correction_bias_.defined() &&
      !e_score_correction_bias_is_loaded_) {
    LOAD_WEIGHT(e_score_correction_bias);
  }
}

void FusedMoEImpl::load_experts(const StateDict& state_dict) {
  for (int idx = 0; idx < num_experts_per_rank_; idx++) {
    auto expert_state_dict =
        state_dict.get_dict_with_prefix(std::to_string(start_expert_id_ + idx));
    load_w13(expert_state_dict.get_dict_with_prefix(".gate_proj."), idx, true);
    load_w13(expert_state_dict.get_dict_with_prefix(".up_proj."), idx, false);
    load_w2(expert_state_dict.get_dict_with_prefix(".down_proj."), idx);
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
