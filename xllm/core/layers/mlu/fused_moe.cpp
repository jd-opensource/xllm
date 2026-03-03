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

#include "fused_moe.h"

#include <glog/logging.h>

#include <iomanip>

#include "common/global_flags.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#include "layers/common/dp_utils.h"
#include "util/tensor_helper.h"
#include "util/utils.h"

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

  // Deep EP initialization check
  enable_deep_ep_ = FLAGS_expert_parallel_degree == 2 && ep_size > 1;
  if (enable_deep_ep_) {
    // for now, we only implement the deep ep for decode stage.
    // so we will assume the max_token_num is limited to max_batch_size * (1+K)
    // K is the number of speculative tokens.
    int64_t dispatch_token_size;
    if (quant_args.quant_method() == "smoothquant") {
      // float32 is for the scale of the quantized input
      dispatch_token_size = hidden_size_ * get_dtype_size(torch::kInt8) +
                            get_dtype_size(torch::kFloat32);
    } else {
      dispatch_token_size =
          hidden_size_ * get_dtype_size(options_.dtype().toScalarType());
    }
    torch::ScalarType combine_dtype = options_.dtype().toScalarType();
    int64_t combine_token_size = hidden_size_ * get_dtype_size(combine_dtype);
    // Ensure calculation base is at least ep_size
    int64_t effective_seqs =
        std::max((int64_t)FLAGS_max_seqs_per_batch, (int64_t)ep_size);
    // NOTE: FLAGS_max_seqs_per_batch represents the maximum total batch size,
    // regardless of the dp size. To ensure robust scheduling and account
    // for the worst-case scenario, we must guarantee that each rank is capable
    // of handling the maximum possible number of tokens. Therefore, we define
    // max_num_tokens_per_rank as the full maximum value, without dividing by
    // either the rank count or the dp size.
    int64_t max_num_tokens_per_rank =
        (1 + FLAGS_num_speculative_tokens) * effective_seqs * topk_;

    // make sure that all layers share the same deep ep instance
    //  so that the memory footprint is minimized
    deep_ep_ = DeepEPManager::get_instance(dispatch_token_size,
                                           combine_token_size,
                                           max_num_tokens_per_rank,
                                           num_experts,
                                           parallel_args,
                                           options_);

    // obtain the buffer and parameters of deep ep
    deep_ep_buffer_ = deep_ep_->get_buffer();
    deep_ep_params_ = deep_ep_->get_params();

    // intermediate buffer that can be initialized once
    // we place these tensor here in order to speed up forward pass
    int64_t n_tokens_recv = deep_ep_params_.max_num_tokens_recv;
    int64_t token_bytes = is_smoothquant_
                              ? get_dtype_size(torch::kInt8)
                              : get_dtype_size(options_.dtype().toScalarType());
    token_bytes = token_bytes * hidden_size_;
    int64_t head_size = n_tokens_recv * token_bytes;
    dispatch_recv_token_tensor_head_ =
        deep_ep_buffer_.combine_send_token_tensor.narrow(0, 0, head_size)
            .view({n_tokens_recv, token_bytes});
    // input scale in smoothquant
    if (is_smoothquant_) {
      int64_t tail_size = n_tokens_recv * get_dtype_size(torch::kFloat32);
      dispatch_recv_token_tensor_tail_ =
          deep_ep_buffer_.combine_send_token_tensor
              .narrow(0, head_size, tail_size)
              .view({n_tokens_recv, -1});
    }
  }

  // calculate the number of experts per rank
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;

  gate_ = register_module("gate", MoEGate(model_args, quant_args, options));
  if (n_shared_experts_ > 0) {
    ProcessGroup* shared_expert_pg;
    if (parallel_args_.ep_size() > 1) {
      // we use tp=1 for shared experts computation in deep ep mode
      CHECK(parallel_args_.ep_size() == parallel_args_.world_size())
          << "Models with shared experts only support ep_size equal to "
             "world size for now.";
      shared_expert_pg = parallel_args.moe_tp_group_;
    } else {
      shared_expert_pg = parallel_args.process_group_;
    }
    // The shared experts computation can proceed in parallel with the
    // final communication step during the MoE computation, as long as it
    // remains independent of any communication operations. For optimal
    // performance, ensure that the shared experts layer on each rank always
    // maintains its own unique weights.
    shared_experts_ =
        register_module("shared_experts",
                        DenseMLP(hidden_size_,
                                 intermediate_size * n_shared_experts_,
                                 is_gated_,
                                 false,
                                 hidden_act_,
                                 /*enable_result_reduction=*/true,
                                 quant_args,
                                 shared_expert_pg,
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
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
            quant_option),
        false);
    w13_scale_ = register_parameter(
        "w13_scale",
        torch::empty({num_experts_per_rank_, local_intermediate_size * 2},
                     fp_option),
        false);
    // Note: We do not check enable_deep_ep_ here, since smooth quantization
    // information may be needed even when deep EP mode is disabled. This allows
    // retrieving quantization parameters for any subset of experts as required.
    input_smooth_ = register_parameter(
        "input_smooth",
        torch::empty({num_total_experts_, hidden_size_}, fp_option),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size},
            quant_option),
        false);
    w2_scale_ = register_parameter(
        "w2_scale",
        torch::empty({num_experts_per_rank_, hidden_size_}, fp_option),
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
            {num_experts_per_rank_, local_intermediate_size * 2, hidden_size_},
            options_),
        false);
    w2_ = register_parameter(
        "w2",
        torch::empty(
            {num_experts_per_rank_, hidden_size_, local_intermediate_size},
            options_),
        false);
  }
}

torch::Tensor FusedMoEImpl::create_group_gemm_output(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& group_list,
    torch::ScalarType dtype,
    torch::Tensor& workspace) {
  // unify shape logic: define the target shape once.
  bool is_3d_weight = (b.dim() != 2);
  int64_t num_tokens = a.size(0);
  int64_t out_dim = is_3d_weight ? b.size(1) : b.size(0);

  std::vector<int64_t> output_shape;
  int64_t required_elements = num_tokens * out_dim;

  if (is_3d_weight) {
    output_shape = {num_tokens, out_dim};
  } else {
    output_shape = {group_list.size(0), num_tokens, out_dim};
    required_elements *= group_list.size(0);
  }

  auto options = a.options().dtype(dtype);

  // non-smoothquant: direct allocation
  if (!is_smoothquant_) {
    return torch::empty(output_shape, options);
  }

  // smoothquant: managed workspace logic
  if (!workspace.defined()) {
    // Lazy initialization: allocate max buffer for the lifecycle
    // Note: accessing class members w13_ and w2_ directly for context
    int64_t max_width = std::max(w13_.size(1), w2_.size(1));
    workspace = torch::empty({num_tokens * max_width}, options);
  }

  // view construction
  CHECK(workspace.numel() >= required_elements)
      << "FusedMoE Workspace too small! Alloc: " << workspace.numel()
      << ", Req: " << required_elements;

  // utilize the pre-calculated output_shape
  return workspace.slice(0, 0, required_elements).view(output_shape);
}

torch::Tensor FusedMoEImpl::forward_experts(const torch::Tensor& hidden_states,
                                            bool enable_all2all_communication) {
  // Dispatcher: route to the appropriate path based on communication mode
  if (enable_all2all_communication) {
    return forward_experts_all2all(hidden_states);
  } else {
    return forward_experts_base(hidden_states);
  }
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                    const ModelInputParams& input_params) {
  // we only support all2all communication for decode stage for now
  bool enable_all2all_communication =
      enable_deep_ep_ && all_dp_ranks_are_decode(input_params);

  bool is_dp_ep_parallel =
      parallel_args_.dp_size() > 1 && parallel_args_.ep_size() > 1;
  // during all2all communication, the output has been
  //  gathered and sliced by dispatch and combine steps,
  //  so we do not need to gather input and slice output again
  bool need_gather_and_slice =
      is_dp_ep_parallel && !enable_all2all_communication;

  auto input = hidden_states;
  if (need_gather_and_slice) {
    input = parallel_state::gather(input,
                                   parallel_args_.dp_local_process_group_,
                                   input_params.dp_global_token_nums);
  }
  auto output = forward_experts(input, enable_all2all_communication);

  if (need_gather_and_slice) {
    output = get_dp_local_slice(output, input_params, parallel_args_);
  }

  return output;
}

void FusedMoEImpl::load_experts(const StateDict& state_dict) {
  const int64_t rank = tp_pg_->rank();
  const int64_t world_size = tp_pg_->world_size();
  const int64_t start_expert_id = start_expert_id_;
  const int64_t num_experts_per_rank = num_experts_per_rank_;
  const int64_t num_total_experts = num_total_experts_;
  std::vector<std::string> prefixes = {"gate_proj.", "up_proj."};
  if (is_smoothquant_) {
    LOAD_MOE_FUSED_WEIGHT("qweight", w1, w3, w13);
    LOAD_MOE_FUSED_WEIGHT("per_channel_scale", w1_scale, w3_scale, w13_scale);
    // When supporting DeepEP All2All mode,
    // we need to load the complete set of expert weights corresponding to
    // "up_proj.smooth". Note that even if deep EP mode is not enabled, it
    // remains possible to retrieve the smooth quantization information for a
    // subset of experts. Therefore, we intentionally do not check whether
    // deep_ep_ is enabled in this case.
    LOAD_MOE_ALL_EXPERT_WEIGHT("up_proj.", "smooth", input_smooth, -1);
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
  load_experts(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm
