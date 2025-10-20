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
#include "kernels/mlu/torch_ops_api.h"
namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(int num_experts,
                           int top_k,
                           int hidden_size,
                           int intermediate_size,
                           bool is_gated,
                           bool has_bias,
                           bool skip_bias_add,
                           int renormalize,
                           const std::string& hidden_act,
                           const std::string& scoring_func,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_experts_(num_experts),
      topk_(top_k),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      is_gated_(is_gated),
      has_bias_(has_bias),
      skip_bias_add_(skip_bias_add),
      renormalize_(renormalize),
      hidden_act_(hidden_act),
      scoring_func_(scoring_func),
      parallel_args_(parallel_args),
      options_(options) {
  int ep_size = parallel_args.ep_size();
  if (parallel_args.moe_tp_group_) {
    ep_local_tp_size_ = parallel_args.moe_tp_group_->world_size();
    ep_local_tp_rank_ = parallel_args.moe_tp_group_->rank();
    ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
    tp_pg_ = parallel_args.moe_tp_group_;
  } else {
    ep_local_tp_size_ = parallel_args.tp_group_->world_size();
    ep_local_tp_rank_ = parallel_args.tp_group_->rank();
    ep_rank_ = parallel_args.rank() / ep_local_tp_size_;
    tp_pg_ = parallel_args.tp_group_;
  }

  num_experts_per_rank_ = num_experts_ / ep_size;
  start_expert_id_ = ep_rank_ * num_experts_per_rank_;

  w13_list_.resize(num_experts_per_rank_);
  w1_list_.resize(num_experts_per_rank_);
  w3_list_.resize(num_experts_per_rank_);
  w2_list_.resize(num_experts_per_rank_);
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const torch::Tensor& router_logits,
                                    std::optional<torch::Tensor> residual) {
  std::optional<torch::Tensor> residual_ = std::nullopt;
  if (residual.has_value() && ep_local_tp_rank_ == 0) {
    residual_ = residual;
  }
  pack_params();
  auto final_hidden_states = xllm::mlu::fused_moe(hidden_states,
                                                  router_logits,
                                                  w13_,
                                                  w2_,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  residual_,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  std::nullopt,
                                                  topk_,
                                                  renormalize_,
                                                  is_gated_,
                                                  hidden_act_,
                                                  scoring_func_,
                                                  start_expert_id_);
  if (tp_pg_->world_size() > 1) {
    final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
  }
  return final_hidden_states;
}

void FusedMoEImpl::pack_params() {
  if (w13_is_loaded_) {
    return;
  }
  w13_ = map_param_data(w13_list_);
  w2_ = map_param_data(w2_list_);
  w13_is_loaded_ = true;

  w13_list_.clear();
  w2_list_.clear();
  w1_list_.clear();
  w3_list_.clear();
}

torch::Tensor FusedMoEImpl::map_param_data(
    const std::vector<torch::Tensor>& param_list) {
  if (param_list.size() == 1) {
    return param_list[0].unsqueeze(0);
  }

  std::vector<torch::Tensor> flattened_params;
  for (const auto& param : param_list) {
    if (!param.defined()) {
      LOG(ERROR)
          << "Parameter(w13/w2) is not defined, please check the state dict.";
      assert(false);
    }
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
  const auto rank = ep_local_tp_rank_;
  const auto world_size = ep_local_tp_size_;
  DEFINE_WEIGHT(weight);
  weight_ = torch::empty({intermediate_size_ / ep_local_tp_size_, hidden_size_},
                         options_);
  LOAD_SHARDED_WEIGHT(weight, 0);
  if (!weight_is_loaded_) {
    return;
  }

  if (is_gated) {
    w1_list_[idx] = weight_;
  } else {
    w3_list_[idx] = weight_;
  }
  if (w1_list_[idx].defined() && w3_list_[idx].defined()) {
    w13_list_[idx] = torch::cat({w1_list_[idx], w3_list_[idx]});
    w1_list_[idx].reset();
    w3_list_[idx].reset();
  }
}

void FusedMoEImpl::load_w2(const StateDict& state_dict, int idx) {
  if (w2_list_[idx].defined() || state_dict.size() == 0) {
    return;
  }
  const auto rank = ep_local_tp_rank_;
  const auto world_size = ep_local_tp_size_;
  DEFINE_WEIGHT(weight);
  weight_ = torch::empty({hidden_size_, intermediate_size_ / ep_local_tp_size_},
                         options_);
  LOAD_SHARDED_WEIGHT(weight, 1);
  if (weight_is_loaded_) {
    w2_list_[idx] = weight_;
  }
}

void FusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  for (int idx = 0; idx < num_experts_per_rank_; idx++) {
    auto expert_state_dict =
        state_dict.get_dict_with_prefix(std::to_string(start_expert_id_ + idx));
    load_w13(expert_state_dict.get_dict_with_prefix(".gate_proj."), idx, true);
    load_w13(expert_state_dict.get_dict_with_prefix(".up_proj."), idx, false);
    load_w2(expert_state_dict.get_dict_with_prefix(".down_proj."), idx);
  }
}

}  // namespace layer
}  // namespace xllm
