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

namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(const ModelArgs& model_args,
                           const FusedMoEArgs& moe_args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options)
    : num_total_experts_(model_args.n_routed_experts()),
      topk_(model_args.num_experts_per_tok()),
      hidden_size_(model_args.hidden_size()),
      n_shared_experts_(model_args.n_shared_experts()),
      is_gated_(moe_args.is_gated),
      hidden_act_(model_args.hidden_act()),
      quant_args_(quant_args),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()) {
  const int64_t num_experts = num_total_experts_;
  const int64_t intermediate_size =
      static_cast<int64_t>(model_args.moe_intermediate_size());
  int64_t ep_size = parallel_args.ep_size();
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
    /*
    The shared_experts are usually implemented using the RowParallelLinear
    layer. Typically, this output serves as the enable_result_reduction results
    for the module. If only tensor parallelism is applied, immediate
    reduction of the shared_experts output isn't necessary; instead, we perform
    the reduction once at the end of the MoE operation.
    */
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
    shared_expert_gate_ = register_module(
        "shared_expert_gate",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, 1).bias(false)));
    shared_expert_gate_->weight.set_data(shared_expert_gate_->weight.to(options));
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

torch::Tensor FusedMoEImpl::select_experts(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d,
    SelectedExpertInfo& selected_expert_info) {
  // prepare the parameters for select_experts
  xllm::kernel::MoeGatingTopkSoftmaxParams gating_topk_params;
  gating_topk_params.x = router_logits_2d;
  gating_topk_params.finished = torch::Tensor();
  gating_topk_params.k = topk_;
  auto [topk_weights, topk_ids, row_ids] = xllm::kernel::moe_gating_topk_softmax(gating_topk_params);
  topk_ids = topk_ids.to(torch::kInt32);
  if (renormalize_) {
    topk_weights = topk_weights / (topk_weights.sum(-1, true) + 1e-6);
  }

  xllm::kernel::MoeInitRoutingV2Params moe_init_routing_params;
  moe_init_routing_params.x = hidden_states_2d;
  moe_init_routing_params.expert_idx = topk_ids;
  moe_init_routing_params.scale = std::nullopt;
  moe_init_routing_params.offset = std::nullopt;
  moe_init_routing_params.active_num = hidden_states_2d.size(0) * topk_;
  moe_init_routing_params.expert_num = num_experts_per_rank_;
  moe_init_routing_params.expert_tokens_num_type = 1;
  moe_init_routing_params.expert_tokens_num_flag = true;
  std::vector<int64_t> expert_range = {start_expert_id_, start_expert_id_ + num_experts_per_rank_};
  moe_init_routing_params.active_expert_range = expert_range;
  moe_init_routing_params.quant_mode = -1;
  auto [expand_hidden_states, expand_row_ids, group_list, dynamic_scale] = xllm::kernel::moe_init_routing_v2(
    moe_init_routing_params);

    // collect the selected tensor
  selected_expert_info.reduce_weight = topk_weights;
  selected_expert_info.combine_idx = expand_row_ids;
  selected_expert_info.token_count_slice = group_list;
  selected_expert_info.cusum_token_count = group_list;
  return expand_hidden_states;
}

torch::Tensor FusedMoEImpl::forward_expert(
    const torch::Tensor& hidden_states,
    const torch::Tensor& router_logits,
    const std::optional<torch::Tensor>& shared_output) {
  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }

  // prepare the parameters for MoE computation
  torch::IntArrayRef hidden_states_shape = hidden_states.sizes();
  torch::ScalarType hidden_states_dtype = hidden_states.dtype().toScalarType();
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)});
  torch::Tensor router_logits_2d =
      router_logits.reshape({-1, router_logits.size(-1)});
  int64_t group_gemm_max_dim = hidden_states_2d.size(0);
  int64_t expert_size = w13_.size(0);

  // Step 1-3: select experts
  SelectedExpertInfo selected_expert_info;
  torch::Tensor expand_hidden_states =
      select_experts(hidden_states_2d, router_logits_2d, selected_expert_info);

  // Step 4: group gemm 1
  torch::Tensor gemm1_out =
      create_group_gemm_output(expand_hidden_states,
                               w13_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype);

  {
    xllm::kernel::GroupedMatmulParams grouped_matmul_params;
    grouped_matmul_params.x = expand_hidden_states;
    if (w13_.size(1) != expand_hidden_states.size(1)) {
      w13_ = w13_.transpose(1, 2);
    }
    grouped_matmul_params.weight = w13_;
    grouped_matmul_params.split_item = 2;
    grouped_matmul_params.group_type = 0;
    grouped_matmul_params.group_list_type = 1;
    grouped_matmul_params.group_list = selected_expert_info.token_count_slice;
    gemm1_out = xllm::kernel::grouped_matmul(grouped_matmul_params).back();
  }

  // Step 5: activation or scaled quantization(fused with activation)
  torch::Tensor act_out;
  torch::Tensor act_out_scale;

  xllm::kernel::ActivationParams activation_params;
  activation_params.input = gemm1_out;
  activation_params.act_mode = "swiglu";
  act_out = xllm::kernel::active_tensor(activation_params);
  // Step 6: group gemm 2
  torch::Tensor gemm2_out =
      create_group_gemm_output(act_out,
                               w2_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype);


  {
    xllm::kernel::GroupedMatmulParams grouped_matmul_params;
    grouped_matmul_params.x = act_out;
    if (w2_.size(1) != act_out.size(1)) {
      w2_ = w2_.transpose(1, 2);
    }
    grouped_matmul_params.weight = w2_;
    grouped_matmul_params.split_item = 2;
    grouped_matmul_params.group_list_type = 1;
    grouped_matmul_params.group_type = 0;
    grouped_matmul_params.group_list = selected_expert_info.token_count_slice;
    gemm2_out = xllm::kernel::grouped_matmul(grouped_matmul_params).back();
  }

  // Step 7: combine the intermediate results and get the final hidden states
  torch::Tensor final_hidden_states;
    xllm::kernel::MoeTokenUnpermuteParams moe_token_unpermute_params;
    moe_token_unpermute_params.permuted_tokens = gemm2_out;
    moe_token_unpermute_params.sorted_indices = selected_expert_info.combine_idx;
    moe_token_unpermute_params.padded_mode = false;
    moe_token_unpermute_params.probes = selected_expert_info.reduce_weight;
    final_hidden_states = xllm::kernel::moe_token_unpermute(moe_token_unpermute_params);
    if (shared_output.has_value()) {
      final_hidden_states = final_hidden_states + shared_output.value();
    }
  // reshape the final hidden states to the original shape
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

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
    if (shared_expert_gate_) {
      auto gate = torch::sigmoid(shared_expert_gate_->forward(input));
      if (shared_output.has_value()) {
        torch::Tensor res = gate * shared_output.value();
        shared_output = res;
      }
    }
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
        state_dict.get_dict_with_prefix("shared_expert."));
    auto weight = state_dict.get_tensor("shared_expert_gate.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(shared_expert_gate_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      shared_expert_gate_->weight.data().copy_(weight);
    }
  }

  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  load_e_score_correction_bias(state_dict.get_dict_with_prefix("gate."));
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
