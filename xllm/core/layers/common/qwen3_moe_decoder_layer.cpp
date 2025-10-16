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

#include "layer_utils.h"

namespace xllm {
namespace layer {

Qwen3MoeDecoderImpl::Qwen3MoeDecoderImpl(const ModelContext& context,
                                         int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  // Initialize attention layers
  attention_ = register_module(
      "self_attn",
      Qwen3Attention(model_args, quant_args, parallel_args, options));

  // Initialize norm layers
  input_norm_ = register_module(
      "input_layernorm",
      RmsNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RmsNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  auto mlp_only_layers = model_args.mlp_only_layers();
  if ((std::count(mlp_only_layers.begin(), mlp_only_layers.end(), layer_id) ==
       0) &&
      model_args.num_experts() > 0 &&
      (layer_id + 1) % model_args.decoder_sparse_step() == 0) {
    moe_mlp_ = register_module("mlp",
                               FusedMoE(model_args.num_experts(),
                                        model_args.num_experts_per_tok(),
                                        model_args.hidden_size(),
                                        model_args.moe_intermediate_size(),
                                        0,
                                        true,
                                        false,
                                        false,
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
                                    quant_args,
                                    parallel_args,
                                    options));
  }
}

void Qwen3MoeDecoderImpl::load_state_dict(const StateDict& state_dict) {
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

torch::Tensor Qwen3MoeDecoderImpl::forward(
    torch::Tensor& x,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Pre-attention norm
  torch::Tensor residual = x;
  x = input_norm_(x);

  // Attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);
  x = x + residual;

  // Post-attention norm
  residual = x;
  x = post_norm_(x);

  // MLP forward
  if (moe_mlp_) {
    x = moe_mlp_(x, input_params);
  } else {
    x = mlp_(x);
  }
  x = x + residual;

  return x;
}

}  // namespace layer
}  // namespace xllm
