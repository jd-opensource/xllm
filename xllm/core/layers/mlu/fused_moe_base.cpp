/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// FusedMoE Base path implementation (AllReduce communication mode)
// This file contains the Base path methods extracted from fused_moe.cpp

#include <glog/logging.h>

#include "framework/parallel_state/parallel_state.h"
#include "fused_moe.h"
#include "kernels/ops_api.h"
#include "util/tensor_helper.h"

namespace xllm {
namespace layer {

torch::Tensor FusedMoEImpl::select_experts_base(
    const torch::Tensor& hidden_states_2d,
    const torch::Tensor& reduce_weight,
    const torch::Tensor& expert_id,
    SelectedExpertInfo& selected_expert_info) {
  int64_t expert_size = w13_.size(0);

  // Step 1: generate expert ids (gate / routing already done by MoEGate)
  torch::Tensor gather_idx;
  torch::Tensor combine_idx;
  torch::Tensor token_count;
  std::optional<torch::Tensor> cusum_token_count;
  {
    xllm::kernel::MoeGenIdxParams moe_gen_idx_params;
    moe_gen_idx_params.expert_id = expert_id;
    moe_gen_idx_params.expert_num = num_total_experts_;
    std::vector<torch::Tensor> output_vec =
        xllm::kernel::moe_gen_idx(moe_gen_idx_params);
    gather_idx = output_vec[0];
    combine_idx = output_vec[1];
    token_count = output_vec[2];
    // Base path uses cusum_token_count
    cusum_token_count = output_vec[3];
  }

  // Step 2: expand and quantize input if needed
  torch::Tensor expand_hidden_states;
  torch::Tensor hidden_states_scale;
  torch::Tensor token_count_slice;

  // Base path: slice the token count for local experts
  token_count_slice =
      token_count.slice(0, start_expert_id_, start_expert_id_ + expert_size);

  if (is_smoothquant_) {
    xllm::kernel::ScaledQuantizeParams scaled_quantize_params;
    scaled_quantize_params.x = hidden_states_2d;
    scaled_quantize_params.smooth = input_smooth_.slice(
        0, start_expert_id_, start_expert_id_ + expert_size);
    scaled_quantize_params.gather_index_start_position =
        cusum_token_count.value().index({start_expert_id_}).unsqueeze(0);
    scaled_quantize_params.token_count = token_count_slice;
    scaled_quantize_params.gather_index = gather_idx;
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

void FusedMoEImpl::final_comm_allreduce(torch::Tensor& final_hidden_states,
                                        const torch::Tensor& hidden_states,
                                        torch::Tensor& shared_expert_output) {
  // Communication Step 3: AllReduce for non-all2all communication
  // shared experts can be parallelized with the final communication step
  // during moe computation.
  auto current_stream = device_.current_stream();
  routed_stream_->wait_stream(*current_stream);
  {
    torch::StreamGuard stream_guard = routed_stream_->set_stream_guard();
    if (tp_pg_->world_size() > 1) {
      final_hidden_states = parallel_state::reduce(final_hidden_states, tp_pg_);
    }
    if (parallel_args_.ep_size() > 1) {
      final_hidden_states = parallel_state::reduce(
          final_hidden_states, parallel_args_.moe_ep_group_);
    }
  }

  if (n_shared_experts_ > 0) {
    shared_stream_->wait_stream(*current_stream);
    torch::StreamGuard stream_guard = shared_stream_->set_stream_guard();
    // for non all2all, we compute the shared experts parallelized with the
    // final communication step
    shared_expert_output = shared_experts_(hidden_states);
    shared_expert_output =
        shared_expert_output.reshape({-1, shared_expert_output.size(-1)});
  }

  // join for parallelization
  current_stream->wait_stream(*routed_stream_);
  if (n_shared_experts_ > 0) {
    current_stream->wait_stream(*shared_stream_);
    final_hidden_states += shared_expert_output;
  }
}

torch::Tensor FusedMoEImpl::forward_experts_base(
    const torch::Tensor& hidden_states) {
  if (!stream_initialized_) {
    device_ = xllm::Device(hidden_states.device());
    routed_stream_ = device_.get_stream_from_pool();
    shared_stream_ = device_.get_stream_from_pool();
    stream_initialized_ = true;
  }

  torch::Tensor shared_expert_output;
  torch::IntArrayRef hidden_states_shape = hidden_states.sizes();
  torch::ScalarType hidden_states_dtype = hidden_states.dtype().toScalarType();
  torch::Tensor hidden_states_2d =
      hidden_states.reshape({-1, hidden_states.size(-1)});

  torch::Tensor reduce_weight;
  torch::Tensor expert_id;
  std::tie(reduce_weight, expert_id) = gate_->forward(hidden_states_2d);

  int64_t group_gemm_max_dim = hidden_states_2d.size(0);
  int64_t expert_size = w13_.size(0);

  SelectedExpertInfo selected_expert_info;
  torch::Tensor expand_hidden_states = select_experts_base(
      hidden_states_2d, reduce_weight, expert_id, selected_expert_info);

  torch::Tensor gemm_workspace;

  // Step 4: group gemm 1
  torch::Tensor gemm1_out =
      create_group_gemm_output(expand_hidden_states,
                               w13_,
                               selected_expert_info.token_count_slice,
                               hidden_states_dtype,
                               gemm_workspace);
  // ensure the lifespan of these parameters via brace
  {
    xllm::kernel::GroupGemmParams group_gemm_params;
    torch::ScalarType a_dtype =
        is_smoothquant_ ? torch::kInt8 : hidden_states_dtype;
    group_gemm_params.a =
        view_as_dtype(expand_hidden_states, a_dtype).view({-1, hidden_size_});
    group_gemm_params.b = w13_;
    group_gemm_params.token_count = selected_expert_info.token_count_slice;
    if (is_smoothquant_) {
      torch::Tensor a_scale =
          selected_expert_info.input_scale.value().flatten();
      selected_expert_info.input_scale =
          view_as_dtype(a_scale, torch::kFloat32);
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
                               hidden_states_dtype,
                               gemm_workspace);
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

  // After group gemm is finished, some tensors are no
  // longer needed. We must explicitly release the memory.
  expand_hidden_states = torch::Tensor();
  selected_expert_info.input_scale = std::nullopt;
  act_out = torch::Tensor();

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

    final_hidden_states =
        xllm::kernel::moe_combine_result(moe_combine_result_params);
  }

  // reshape the final hidden states to the original shape
  final_hidden_states = final_hidden_states.reshape(hidden_states_shape);

  // Communication Step 3: AllReduce
  final_comm_allreduce(
      final_hidden_states, hidden_states, shared_expert_output);

  return final_hidden_states;
}

}  // namespace layer
}  // namespace xllm
