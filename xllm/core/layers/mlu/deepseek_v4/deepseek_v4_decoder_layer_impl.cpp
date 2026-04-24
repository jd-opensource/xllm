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

#include "layers/mlu/deepseek_v4/deepseek_v4_decoder_layer_impl.h"

#include "common/global_flags.h"

namespace xllm {
namespace layer {

DeepSeekV4DecoderLayerImpl::DeepSeekV4DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id,
    int64_t cached_state_num)
    : layer_id_(layer_id), parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();

  // Validate parallel strategy: only pure TP and EP are supported
  // DP (Data Parallel) is not supported in this implementation
  CHECK(parallel_args_.dp_size() == 1)
      << "DeepSeek V4 decoder layer only supports pure TP and EP parallel "
         "strategies. DP is not supported (dp_size must be 1, got "
      << parallel_args_.dp_size() << ")";

  // EP must equal world_size when expert parallel is enabled
  if (parallel_args_.ep_size() > 1) {
    CHECK(parallel_args_.ep_size() == parallel_args_.world_size())
        << "DeepSeek V4 MoE only supports ep_size equal to world size when "
           "EP is enabled (ep_size="
        << parallel_args_.ep_size()
        << ", world_size=" << parallel_args_.world_size() << ")";
  }

  // Store model configuration
  hidden_size_ = model_args.hidden_size();
  norm_eps_ = model_args.rms_norm_eps();

  // HyperConnection parameters
  hc_mult_ = model_args.hc_mult();
  hc_sinkhorn_iters_ = model_args.hc_sinkhorn_iters();
  hc_eps_ = model_args.hc_eps();

  // Initialize HyperConnection modules for attention
  hc_attn_pre_ = register_module("hc_attn_pre",
                                 HyperConnectionPre(hc_mult_,
                                                    hidden_size_,
                                                    hc_sinkhorn_iters_,
                                                    hc_eps_,
                                                    norm_eps_,
                                                    options));
  hc_attn_post_ = register_module("hc_attn_post", HyperConnectionPost(options));

  // Initialize HyperConnection modules for MoE
  hc_moe_pre_ = register_module("hc_moe_pre",
                                HyperConnectionPre(hc_mult_,
                                                   hidden_size_,
                                                   hc_sinkhorn_iters_,
                                                   hc_eps_,
                                                   norm_eps_,
                                                   options));
  hc_moe_post_ = register_module("hc_moe_post", HyperConnectionPost(options));

  // Initialize attention normalization
  attn_norm_ =
      register_module("attn_norm", RMSNorm(hidden_size_, norm_eps_, options));

  // Initialize attention layer
  attention_ = register_module("attn",
                               DeepSeekV4Attention(model_args,
                                                   quant_args,
                                                   parallel_args_,
                                                   options,
                                                   layer_id,
                                                   cached_state_num));

  // Initialize MoE normalization
  moe_norm_ =
      register_module("ffn_norm", RMSNorm(hidden_size_, norm_eps_, options));

  // Initialize MoE FFN layer
  // Determine use_hash based on layer_id and n_hash_layers
  // Hash-based routing is used for layers where layer_id < n_hash_layers
  use_hash_ = layer_id < model_args.n_hash_layers();
  route_gate_ = register_module("route_gate",
                                ReplicatedLinear(model_args.hidden_size(),
                                                 model_args.n_routed_experts(),
                                                 /*bias=*/false,
                                                 quant_args,
                                                 options));
  topk_ = register_module("topk",
                          DeepSeekV4TopK(model_args.n_routed_experts(),
                                         model_args.num_experts_per_tok(),
                                         model_args.routed_scaling_factor(),
                                         model_args.vocab_size(),
                                         use_hash_,
                                         options));

  const FusedMoEArgs moe_args{.is_gated = true,
                              .enable_result_reduction = true};
  moe_mlp_ = register_module(
      "moe",
      FusedMoE(model_args, moe_args, quant_args, parallel_args_, options));
}

void DeepSeekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  torch::NoGradGuard no_grad;

  // Load HyperConnection weights
  hc_attn_pre_->load_state_dict(
      state_dict.get_dict_with_prefix("hc_attn_pre."));
  hc_moe_pre_->load_state_dict(state_dict.get_dict_with_prefix("hc_ffn_pre."));

  // Load normalization weights
  attn_norm_->load_state_dict(state_dict.get_dict_with_prefix("attn_norm."));
  moe_norm_->load_state_dict(state_dict.get_dict_with_prefix("ffn_norm."));

  // Load attention weights
  attention_->load_state_dict(state_dict.get_dict_with_prefix("attn."));

  // Load MoE MLP weights
  route_gate_->load_state_dict(state_dict.get_dict_with_prefix("ffn.gate."));
  topk_->load_state_dict(state_dict.get_dict_with_prefix("ffn.gate."));
  moe_mlp_->load_state_dict(state_dict.get_dict_with_prefix("ffn."));
}

torch::Tensor DeepSeekV4DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::vector<int64_t>& batch_to_kv_state,
    const std::optional<torch::Tensor>& input_ids) {
  static_cast<void>(input_params);
  // Save residual for HyperConnection
  residual = x;

  // HyperConnection pre-processing for attention
  PrePostCombOutput hc_attn_output = hc_attn_pre_->forward(x);
  x = hc_attn_output.pre;
  torch::Tensor post = hc_attn_output.post;
  torch::Tensor comb = hc_attn_output.comb;

  // Pre-attention normalization
  x = std::get<0>(attn_norm_->forward(x));

  // Attention forward
  x = attention_->forward(
      positions, x, attn_metadata, kv_cache, batch_to_kv_state);

  // HyperConnection post-processing for attention
  x = hc_attn_post_->forward(x, residual.value(), post, comb);

  // Save residual for HyperConnection
  residual = x;

  // HyperConnection pre-processing for MoE
  PrePostCombOutput hc_moe_output = hc_moe_pre_->forward(x);

  x = hc_moe_output.pre;
  post = hc_moe_output.post;
  comb = hc_moe_output.comb;

  // Pre-MoE normalization
  x = std::get<0>(moe_norm_->forward(x));

  torch::Tensor route_x = x.reshape({-1, x.size(-1)});
  torch::Tensor scores = route_gate_->forward(route_x);

  std::optional<torch::Tensor> route_ids = std::nullopt;
  if (use_hash_) {
    CHECK(input_ids.has_value()) << "input_ids is required for hash routing";
    torch::Tensor ids = input_ids.value().to(torch::kInt64);
    CHECK_EQ(scores.size(0) % ids.size(0), 0)
        << "routing token count must be a multiple of input_ids count";
    const int64_t repeat = scores.size(0) / ids.size(0);
    route_ids = repeat == 1 ? ids : ids.repeat_interleave(repeat);
  }

  DeepSeekV4TopKOutput route_output = topk_->forward(scores, route_ids);
  FusedMoEImpl::RouteInfo route_info;
  route_info.reduce_weight = route_output.weights;
  route_info.expert_id = route_output.indices;

  x = moe_mlp_->forward_experts(x,
                                /*enable_all2all_communication=*/false,
                                route_info);

  // HyperConnection post-processing for MoE
  x = hc_moe_post_->forward(x, residual.value(), post, comb);

  return x;
}

}  // namespace layer
}  // namespace xllm
