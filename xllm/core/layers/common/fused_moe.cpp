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
                                 /*enable_result_reduction=*/false,
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

torch::Tensor FusedMoEImpl::select_experts(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& router_logits_2d,
    SelectedExpertInfo& selected_expert_info) {
  // prepare the parameters for select_experts
  std::optional<torch::Tensor> e_score_correction_bias = std::nullopt;
  if (e_score_correction_bias_.defined()) {
    e_score_correction_bias = e_score_correction_bias_;
  }
  int64_t expert_size = w13_.size(0);

  // Step 1: apply softmax topk or sigmoid topk / routing logic
  torch::Tensor reduce_weight;
  torch::Tensor expert_id;
  {
    xllm::kernel::MoeActiveTopkParams moe_active_topk_params;
    moe_active_topk_params.input = router_logits_2d;
    moe_active_topk_params.topk = topk_;
    moe_active_topk_params.num_expert_group = num_expert_group_;
    moe_active_topk_params.topk_group = topk_group_;
    moe_active_topk_params.normalize = renormalize_;
    moe_active_topk_params.normed_by = "topk_logit";
    moe_active_topk_params.scoring_func = scoring_func_;
    moe_active_topk_params.route_scale = route_scale_;
    moe_active_topk_params.e_score_correction_bias = e_score_correction_bias;
    std::tie(reduce_weight, expert_id) =
        xllm::kernel::moe_active_topk(moe_active_topk_params);
  }

  // Step 2: generate expert ids
  torch::Tensor gather_idx;
  torch::Tensor combine_idx;
  torch::Tensor token_count;
  torch::Tensor cusum_token_count;
  {
    xllm::kernel::MoeGenIdxParams moe_gen_idx_params;
    moe_gen_idx_params.expert_id = expert_id;
    moe_gen_idx_params.expert_num = router_logits_2d.size(-1);
    std::vector<torch::Tensor> output_vec =
        xllm::kernel::moe_gen_idx(moe_gen_idx_params);
    gather_idx = output_vec[0];
    combine_idx = output_vec[1];
    token_count = output_vec[2];
    cusum_token_count = output_vec[3];
  }

  // Step 3: expand and quantize input if needed
  torch::Tensor expand_hidden_states;
  torch::Tensor hidden_states_scale;
  torch::Tensor token_count_slice =
      token_count.slice(0, start_expert_id_, start_expert_id_ + expert_size);
  if (is_smoothquant_) {
    xllm::kernel::ScaledQuantizeParams scaled_quantize_params;
    scaled_quantize_params.x = hidden_states_2d;
    scaled_quantize_params.smooth = input_smooth_;
    scaled_quantize_params.token_count = token_count_slice;
    scaled_quantize_params.gather_index = gather_idx;
    scaled_quantize_params.gather_index_start_position =
        cusum_token_count.index({start_expert_id_}).unsqueeze(0);
    scaled_quantize_params.act_mode = "none";
    scaled_quantize_params.active_coef = 1.0;
    scaled_quantize_params.is_gated = false;
    scaled_quantize_params.quant_type = torch::kChar;
    std::tie(expand_hidden_states, hidden_states_scale) =
        xllm::kernel::scaled_quantize(scaled_quantize_params);
  } else {
    xllm::kernel::MoeExpandInputParams moe_expand_input_params;
    moe_expand_input_params.input = hidden_states_2d;
    moe_expand_input_params.gather_index = gather_idx;
    moe_expand_input_params.cusum_token_count = cusum_token_count;
    moe_expand_input_params.start_expert_id = start_expert_id_;
    moe_expand_input_params.expert_size = expert_size;
    expand_hidden_states =
        xllm::kernel::moe_expand_input(moe_expand_input_params);
  }

  // collect the selected tensor
  selected_expert_info.reduce_weight = reduce_weight;
  selected_expert_info.combine_idx = combine_idx;
  selected_expert_info.token_count_slice = token_count_slice;
  selected_expert_info.cusum_token_count = cusum_token_count;
  if (is_smoothquant_) {
    selected_expert_info.input_scale = hidden_states_scale;
  }

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
  // ensure the lifespan of these parameters via brace
  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    group_gemm_params.a = expand_hidden_states;
    group_gemm_params.b = w13_;
    group_gemm_params.token_count = selected_expert_info.token_count_slice;
    if (is_smoothquant_) {
      group_gemm_params.a_scale = selected_expert_info.input_scale;
      group_gemm_params.b_scale = w13_scale_;
    }
    group_gemm_params.max_dim = group_gemm_max_dim;
    group_gemm_params.trans_a = false;
    group_gemm_params.trans_b = true;
    group_gemm_params.a_quant_bit = is_smoothquant_ ? 8 : -1;
    group_gemm_params.output = gemm1_out;
    gemm1_out = xllm::kernel::group_gemm(group_gemm_params);
  }

  // Step 5: activation or scaled quantization(fused with activation)
  torch::Tensor act_out;
  torch::Tensor act_out_scale;
  if (is_smoothquant_) {
    int64_t slice_dim = gemm1_out.size(1);
    if (is_gated_) slice_dim /= 2;
    // slice operation is a view, does not take up extra memory, but points to
    // the same memory
    act_out = expand_hidden_states.slice(1, 0, slice_dim);
    act_out_scale =
        selected_expert_info.input_scale.value().slice(0, 0, gemm1_out.size(0));
    // call scaled quantization kernel (also fused with activation)
    xllm::kernel::ScaledQuantizeParams scaled_quantize_params;
    scaled_quantize_params.x = gemm1_out;
    scaled_quantize_params.smooth = act_smooth_;
    scaled_quantize_params.token_count = selected_expert_info.token_count_slice;
    scaled_quantize_params.output = act_out;
    scaled_quantize_params.output_scale = act_out_scale;
    scaled_quantize_params.act_mode = hidden_act_;
    scaled_quantize_params.active_coef = 1.0;
    scaled_quantize_params.is_gated = is_gated_;
    scaled_quantize_params.quant_type = torch::kChar;
    std::tie(act_out, act_out_scale) =
        xllm::kernel::scaled_quantize(scaled_quantize_params);
  } else {
    act_out =
        is_gated_ ? gemm1_out.slice(1, 0, gemm1_out.size(1) / 2) : gemm1_out;
    // call activation kernel
    xllm::kernel::ActivationParams activation_params;
    activation_params.input = gemm1_out;
    activation_params.output = act_out;
    activation_params.cusum_token_count =
        selected_expert_info.cusum_token_count;
    activation_params.act_mode = hidden_act_;
    activation_params.is_gated = is_gated_;
    activation_params.start_expert_id = start_expert_id_;
    activation_params.expert_size = expert_size;
    xllm::kernel::active(activation_params);
  }

  // Step 6: group gemm 2
  torch::Tensor gemm2_out =
      create_group_gemm_output(act_out,
                               w2_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype);
  // ensure the lifespan of these parameters via brace
  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    group_gemm_params.a = act_out;
    group_gemm_params.b = w2_;
    group_gemm_params.token_count = selected_expert_info.token_count_slice;
    if (is_smoothquant_) {
      group_gemm_params.a_scale = act_out_scale;
      group_gemm_params.b_scale = w2_scale_;
    }
    group_gemm_params.max_dim = group_gemm_max_dim;
    group_gemm_params.trans_a = false;
    group_gemm_params.trans_b = true;
    group_gemm_params.a_quant_bit = is_smoothquant_ ? 8 : -1;
    group_gemm_params.output = gemm2_out;
    gemm2_out = xllm::kernel::group_gemm(group_gemm_params);
  }
  // After group gemm is finished, expand_hidden_states and input_scale are no
  // longer needed. We must explicitly release the memory.
  expand_hidden_states = torch::Tensor();
  selected_expert_info.input_scale = std::nullopt;

  // Step 7: combine the intermediate results and get the final hidden states
  torch::Tensor final_hidden_states;
  // ensure the lifespan of these parameters via brace
  {
    xllm::kernel::MoeCombineResultParams moe_combine_result_params;
    moe_combine_result_params.input = gemm2_out;
    moe_combine_result_params.reduce_weight =
        selected_expert_info.reduce_weight;
    moe_combine_result_params.gather_ids = selected_expert_info.combine_idx;
    moe_combine_result_params.cusum_token_count =
        selected_expert_info.cusum_token_count;
    moe_combine_result_params.start_expert_id = start_expert_id_;
    moe_combine_result_params.expert_size = expert_size;
    moe_combine_result_params.bias = std::nullopt;
    // make sure residual fits the requirements of moe_combine_result
    if (shared_output.has_value()) {
      moe_combine_result_params.residual =
          shared_output.value().reshape({-1, shared_output.value().size(-1)});
    }
    final_hidden_states =
        xllm::kernel::moe_combine_result(moe_combine_result_params);
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
