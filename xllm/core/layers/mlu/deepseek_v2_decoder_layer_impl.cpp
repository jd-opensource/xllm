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

namespace xllm {
namespace layer {

DeepseekV2DecoderLayerImpl::DeepseekV2DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();

  // get rank and world_size from parallel_args_
  rank_ = parallel_args_.rank();
  world_size_ = parallel_args_.world_size();

  // Initialize attention layers
  attention_ = register_module(
      "self_attn",
      DeepseekV2Attention(model_args, quant_args, parallel_args_, options));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  auto first_k_dense_replace = model_args.first_k_dense_replace();
  if (layer_id >= first_k_dense_replace) {
    moe_mlp_ = register_module("mlp",
                               FusedMoE(model_args.n_routed_experts(),
                                        model_args.num_experts_per_tok(),
                                        model_args.n_group(),
                                        model_args.topk_group(),
                                        model_args.routed_scaling_factor(),
                                        model_args.hidden_size(),
                                        model_args.moe_intermediate_size(),
                                        model_args.n_shared_experts(),
                                        /*is_gated=*/true,
                                        /*has_score_bias=*/false,
                                        /*has_bias=*/false,
                                        /*skip_bias_add=*/false,
                                        model_args.norm_topk_prob(),
                                        model_args.hidden_act(),
                                        model_args.scoring_func(),
                                        model_args.topk_method(),
                                        quant_args,
                                        parallel_args_,
                                        options));
  } else {
    mlp_ = register_module("mlp",
                           DenseMLP(model_args.hidden_size(),
                                    model_args.intermediate_size(),
                                    /*is_gated=*/true,
                                    /*has_bias=*/false,
                                    model_args.hidden_act(),
                                    /*enable_result_reduction=*/true,
                                    quant_args,
                                    parallel_args_,
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

torch::Tensor DeepseekV2DecoderLayerImpl::forward(
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

  // add tensor model group all reduce
  // to avoid implicit communcation in deepseek attention layer.
  if (world_size_ > 1) {
    x = xllm::parallel_state::reduce(x, parallel_args_.tp_group_);
  }

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
