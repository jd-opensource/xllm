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

#include "qwen3_moe_decoder_layer.h"

#include <glog/logging.h>

#include "common/global_flags.h"

namespace xllm {
namespace layer {

Qwen3MoeDecoderLayerImpl::Qwen3MoeDecoderLayerImpl(const ModelContext& context,
                                                   int32_t layer_id) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  // Qwen3 only support deep ep all2all
  //  when dp_size == ep_size && dp_size == world_size for now
  bool enable_deep_ep = FLAGS_expert_parallel_degree == 2;
  if (enable_deep_ep) {
    CHECK_EQ(parallel_args.dp_size(), parallel_args.world_size())
        << "Qwen3 MoE only support deep ep all2all when dp_size == world_size";
    CHECK_EQ(parallel_args.dp_size(), parallel_args.ep_size())
        << "Qwen3 MoE only support deep ep all2all when dp_size == ep_size";
  }

  // Initialize attention layers
  attention_ = register_module("self_attn", Qwen2Attention(context));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  auto mlp_only_layers = model_args.mlp_only_layers();
  if ((std::count(mlp_only_layers.begin(), mlp_only_layers.end(), layer_id) ==
       0) &&
      model_args.num_experts() > 0 &&
      (layer_id + 1) % model_args.decoder_sparse_step() == 0) {
    moe_mlp_ = register_module("mlp",
                               FusedMoE(model_args.num_experts(),
                                        model_args.num_experts_per_tok(),
                                        -1,   // num_expert_group
                                        0,    // topk_group
                                        1.0,  // route_scale
                                        model_args.hidden_size(),
                                        model_args.moe_intermediate_size(),
                                        0,      // n_shared_experts
                                        true,   // is_gated
                                        false,  // has_score_bias
                                        false,  // has_bias
                                        false,  // skip_bias_add
                                        model_args.norm_topk_prob(),
                                        model_args.hidden_act(),
                                        "softmax",
                                        "",
                                        quant_args,
                                        parallel_args,
                                        options));
  } else {
    mlp_ = register_module("mlp",
                           DenseMLP(model_args.hidden_size(),
                                    model_args.intermediate_size(),
                                    true,
                                    false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/true,
                                    quant_args,
                                    parallel_args.tp_group_,
                                    options));
  }
}

void Qwen3MoeDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
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

torch::Tensor Qwen3MoeDecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Pre-attention norm
  if (!residual.has_value()) {
    residual = x;
    x = std::get<0>(input_norm_->forward(x));
  } else {
    std::tie(x, residual) = input_norm_->forward(x, residual);
  }

  // Attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);

  // Post-attention norm
  std::tie(x, residual) = post_norm_->forward(x, residual);

  // MLP forward
  if (moe_mlp_) {
    x = moe_mlp_(x, input_params);
  } else {
    x = mlp_(x);
  }

  return x;
}

}  // namespace layer
}  // namespace xllm
