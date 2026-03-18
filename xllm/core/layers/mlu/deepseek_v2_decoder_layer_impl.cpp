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

#include "deepseek_v2_decoder_layer_impl.h"

#include "common/global_flags.h"
#include "layers/common/dp_utils.h"

namespace xllm {
namespace layer {

namespace {

bool use_moe_all2all(bool enable_deep_ep,
                     const ModelInputParams& input_params) {
  return enable_deep_ep && all_dp_ranks_are_decode(input_params);
}

torch::Tensor slice_tp_tokens(torch::Tensor x, const ParallelArgs& args);

bool same_sp_topology(ProcessGroup* pg, ProcessGroup* sp_pg) {
  return pg != nullptr && sp_pg != nullptr &&
         pg->world_size() == sp_pg->world_size() && pg->rank() == sp_pg->rank();
}

bool is_sp_alias_pg(ProcessGroup* pg, const ParallelArgs& parallel_args) {
  return pg == parallel_args.process_group_ ||
         pg == parallel_args.moe_ep_group_;
}

std::pair<torch::Tensor, PaddingInfo> shard_attn_out(
    torch::Tensor x,
    const torch::Tensor& residual,
    int64_t target_tokens,
    const ParallelArgs& args,
    DeepseekV2AttentionImpl::PostAttnLayout attn_layout) {
  if (attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kTpShard) {
    return reduce_scatter_attn_input(x, residual, target_tokens, args);
  }

  CHECK(attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kReplicated)
      << "unexpected post-attention layout for TP sharding path: "
      << static_cast<int>(attn_layout);
  x = x + residual;
  auto pad_result = pad_tokens(x, target_tokens);
  return {slice_tp_tokens(pad_result.first, args), pad_result.second};
}

torch::Tensor slice_tp_tokens(torch::Tensor x, const ParallelArgs& args) {
  if (!args.tp_group_ || args.tp_group_->world_size() <= 1) {
    return x;
  }

  int64_t tp_size = args.tp_group_->world_size();
  CHECK_EQ(x.size(0) % tp_size, 0)
      << "token_num " << x.size(0) << " must be divisible by tp_size "
      << tp_size;
  int64_t shard_tokens = x.size(0) / tp_size;
  int64_t start = args.tp_group_->rank() * shard_tokens;
  return x.slice(0, start, start + shard_tokens);
}

}  // namespace

DeepseekV2DecoderLayerImpl::DeepseekV2DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();
  is_moe_layer_ = layer_id >= model_args.first_k_dense_replace();

  // DeepSeek MoE only support ep == world_size when expert parallel is on
  if (parallel_args_.ep_size() > 1) {
    CHECK(parallel_args_.ep_size() == parallel_args_.world_size())
        << "DeepSeek MoE only supports ep_size equal to world size";
  }

  // Keep the all2all enable condition aligned with FusedMoE.
  enable_deep_ep_ = FLAGS_expert_parallel_degree == 2 && is_moe_layer_ &&
                    parallel_args_.ep_size() > 1;

  // Initialize attention layers
  OptimizationConfig optimization_config = context.get_optimization_config();
  attention_ = register_module("self_attn",
                               DeepseekV2Attention(model_args,
                                                   quant_args,
                                                   parallel_args_,
                                                   options,
                                                   optimization_config));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  if (is_moe_layer_) {
    const FusedMoEArgs moe_args{.is_gated = true,
                                .enable_result_reduction = false};
    moe_mlp_ = register_module(
        "mlp",
        FusedMoE(model_args, moe_args, quant_args, parallel_args_, options));
  } else {
    mlp_ = register_module("mlp",
                           DenseMLP(model_args.hidden_size(),
                                    model_args.intermediate_size(),
                                    /*is_gated=*/true,
                                    /*has_bias=*/false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/false,
                                    quant_args,
                                    parallel_args_.tp_group_,
                                    options));
  }
}

void DeepseekV2DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  if (moe_mlp_) {
    moe_mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  } else {
    mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
  }
}

void DeepseekV2DecoderLayerImpl::verify_loaded_weights() const {
  if (moe_mlp_) {
    moe_mlp_->verify_loaded_weights();
  }
}

DeepseekV2DecoderLayerImpl::PostAttnCarrier
DeepseekV2DecoderLayerImpl::build_post_attn_carrier(
    torch::Tensor x,
    const torch::Tensor& residual,
    const ModelInputParams& input_params,
    DeepseekV2AttentionImpl::PostAttnLayout attn_layout,
    bool need_dp_gather,
    bool enable_moe_all2all) {
  PostAttnCarrier carrier;
  if (attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kPackedLocal) {
    CHECK(sequence_parallel_context_ != nullptr)
        << "sequence parallel carrier requires sequence parallel context";
    CHECK(!need_dp_gather)
        << "sequence parallel output path does not support dp gather";
    auto [ffn_in, skip_local] = post_norm_->forward(x, residual);
    carrier.ffn_in = ffn_in;
    carrier.skip_local = skip_local.value();
    carrier.ffn_in = v32_sp::all_gather_across_ranks(
        carrier.ffn_in, *sequence_parallel_context_);
    carrier.mode = PostAttnMode::kPackedLocal;
    return carrier;
  }

  if (enable_moe_all2all) {
    auto shard_result =
        shard_attn_out(x,
                       residual,
                       get_reduce_scatter_tokens(x.size(0), parallel_args_),
                       parallel_args_,
                       attn_layout);
    carrier.ffn_in = shard_result.first;
    carrier.skip_local = carrier.ffn_in;
    carrier.pad_info = shard_result.second;
    carrier.mode = PostAttnMode::kTpPadded;
    return carrier;
  }

  if (need_dp_gather) {
    auto shard_result = shard_attn_out(
        x,
        residual,
        get_dp_gather_tokens(input_params.dp_global_token_nums, parallel_args_),
        parallel_args_,
        attn_layout);
    carrier.ffn_in = shard_result.first;
    carrier.pad_info = shard_result.second;
    carrier.mode = PostAttnMode::kDpGather;

    torch::Tensor local_tokens = carrier.ffn_in;
    if (parallel_args_.tp_group_ &&
        parallel_args_.tp_group_->world_size() > 1) {
      local_tokens =
          parallel_state::gather(local_tokens, parallel_args_.tp_group_, 0);
    }

    CHECK(parallel_args_.dp_local_process_group_ != nullptr)
        << "dp gather carrier requires dp_local_process_group_";
    const int64_t dp_rank = parallel_args_.dp_local_process_group_->rank();
    CHECK_GE(dp_rank, 0) << "invalid dp rank " << dp_rank;
    CHECK_LT(dp_rank,
             static_cast<int64_t>(input_params.dp_global_token_nums.size()))
        << "dp rank " << dp_rank << " exceeds dp_global_token_nums size "
        << input_params.dp_global_token_nums.size();
    const int64_t local_token_num = input_params.dp_global_token_nums[dp_rank];
    carrier.skip_local = local_tokens.slice(0, 0, local_token_num);
    return carrier;
  }

  if (attn_layout == DeepseekV2AttentionImpl::PostAttnLayout::kTpShard) {
    x = xllm::parallel_state::reduce(x, parallel_args_.tp_group_);
  }
  x = x + residual;

  carrier.ffn_in = x;
  carrier.skip_local = x;
  return carrier;
}

torch::Tensor DeepseekV2DecoderLayerImpl::materialize_ffn_input(
    const PostAttnCarrier& carrier,
    const ModelInputParams& input_params) {
  if (carrier.mode != PostAttnMode::kDpGather) {
    return carrier.ffn_in;
  }

  return gather_global_tokens(
      carrier.ffn_in, input_params.dp_global_token_nums, parallel_args_);
}

bool DeepseekV2DecoderLayerImpl::can_keep_local_output(
    const PostAttnCarrier& carrier,
    ProcessGroup* pg) const {
  const bool can_use_sp_fast = carrier.mode == PostAttnMode::kPackedLocal &&
                               sequence_parallel_context_ != nullptr &&
                               sequence_parallel_context_->comm_plan.ffn_can_rs;
  if (!can_use_sp_fast) {
    return false;
  }

  ProcessGroup* const sp_pg = sequence_parallel_context_->process_group;
  if (!pg || pg->world_size() <= 1 || pg == sp_pg) {
    return true;
  }

  if (parallel_args_.dp_size() != 1 || !is_sp_alias_pg(pg, parallel_args_)) {
    return false;
  }

  return same_sp_topology(pg, sp_pg);
}

torch::Tensor DeepseekV2DecoderLayerImpl::comm_out(
    torch::Tensor x,
    const PostAttnCarrier& carrier,
    ProcessGroup* pg) const {
  if (!can_keep_local_output(carrier, pg)) {
    return reduce_out(x, pg);
  }

  if (pg && pg->world_size() > 1) {
    return parallel_state::reduce_scatter(x, pg);
  }

  CHECK(sequence_parallel_context_ != nullptr)
      << "sequence parallel fast path requires sequence parallel context";
  return v32_sp::slice_local_packed(x, *sequence_parallel_context_);
}

torch::Tensor DeepseekV2DecoderLayerImpl::restore_ffn_output(
    torch::Tensor x,
    const PostAttnCarrier& carrier,
    const ModelInputParams& input_params) {
  torch::Tensor skip_local = carrier.skip_local;
  if (carrier.mode == PostAttnMode::kPackedLocal) {
    CHECK(sequence_parallel_context_ != nullptr)
        << "packed restore requires sequence parallel context";
    x = v32_sp::slice_local_packed(x, *sequence_parallel_context_);
    return x + skip_local;
  }

  if (carrier.mode == PostAttnMode::kDpGather) {
    x = get_dp_local_slice(x, input_params, parallel_args_);
    return x + skip_local;
  }

  if (carrier.mode == PostAttnMode::kTpPadded && parallel_args_.tp_group_ &&
      parallel_args_.tp_group_->world_size() > 1) {
    x = parallel_state::gather(x, parallel_args_.tp_group_, 0);
    x = unpad_tokens(x, carrier.pad_info);
    skip_local =
        parallel_state::gather(skip_local, parallel_args_.tp_group_, 0);
    skip_local = unpad_tokens(skip_local, carrier.pad_info);
  }
  return x + skip_local;
}

torch::Tensor DeepseekV2DecoderLayerImpl::reduce_out(torch::Tensor x,
                                                     ProcessGroup* pg) const {
  if (!pg || pg->world_size() <= 1) {
    return x;
  }
  return parallel_state::reduce(x, pg);
}

torch::Tensor DeepseekV2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // we only support all2all communcation for decode stage for now.
  bool enable_moe_all2all = use_moe_all2all(enable_deep_ep_, input_params);
  bool need_dp_gather =
      moe_mlp_ && need_dp_moe_gather(parallel_args_, enable_moe_all2all);

  // Pre-attention norm
  residual = x;
  x = std::get<0>(input_norm_->forward(x));

  // Attention
  x = attention_->forward(
      positions, x, attn_metadata, kv_cache, sequence_parallel_context_);
  const bool use_sp_output =
      sequence_parallel_context_ != nullptr && attention_->can_use_sp();
  const auto attn_layout = attention_->post_attn_layout(use_sp_output);

  // We materialize the carrier immediately after attention so all post-attn
  // communication paths flow through the same norm / ffn / restore stages.
  auto carrier = build_post_attn_carrier(x,
                                         residual.value(),
                                         input_params,
                                         attn_layout,
                                         need_dp_gather,
                                         enable_moe_all2all);
  x = carrier.ffn_in;

  if (carrier.mode != PostAttnMode::kPackedLocal) {
    x = std::get<0>(post_norm_->forward(x));
    carrier.ffn_in = x;
  }
  x = materialize_ffn_input(carrier, input_params);

  // MLP forward
  bool keep_local_output = false;
  auto comm_ffn = [&](torch::Tensor y, ProcessGroup* pg) {
    return keep_local_output ? comm_out(y, carrier, pg) : reduce_out(y, pg);
  };
  if (moe_mlp_) {
    ProcessGroup* routed_pg = parallel_args_.ep_size() > 1
                                  ? parallel_args_.moe_ep_group_
                                  : parallel_args_.tp_group_;
    keep_local_output = !enable_moe_all2all &&
                        can_keep_local_output(carrier, routed_pg) &&
                        can_keep_local_output(carrier, moe_mlp_->shared_pg());
    torch::Tensor shared_out;
    if (!enable_moe_all2all) {
      shared_out = moe_mlp_->forward_shared(x);
      if (shared_out.defined()) {
        shared_out = comm_ffn(shared_out, moe_mlp_->shared_pg());
      }
    }
    x = moe_mlp_->forward_experts(x, enable_moe_all2all);
    if (!enable_moe_all2all) {
      x = comm_ffn(x, routed_pg);
      if (shared_out.defined()) {
        x = x + shared_out;
      }
    }
  } else {
    keep_local_output =
        can_keep_local_output(carrier, parallel_args_.tp_group_);
    x = mlp_(x);
    x = comm_ffn(x, parallel_args_.tp_group_);
  }
  x = keep_local_output ? x + carrier.skip_local
                        : restore_ffn_output(x, carrier, input_params);

  residual = std::nullopt;
  return x;
}

}  // namespace layer
}  // namespace xllm
