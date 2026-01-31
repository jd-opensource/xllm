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

#include "qwen2_decoder_layer.h"

namespace xllm {
namespace layer {

Qwen2DecoderLayerImpl::Qwen2DecoderLayerImpl(const ModelContext& context)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

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

void Qwen2DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  attention_->load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_attention_layernorm."));
  mlp_->load_state_dict(state_dict.get_dict_with_prefix("mlp."));
}

torch::Tensor Qwen2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // Pre-attention norm: Qwen2 norms input only (no residual add). Reset
  // residual to layer input each time so post_norm/MLP use correct residual.
  residual = x;
  // Ensure input dtype matches weight dtype before calling forward
  auto input_norm_weight_dtype = input_norm_->weight().scalar_type();
  auto x_original_dtype = x.scalar_type();
  torch::Tensor x_for_norm = x;
  if (x.scalar_type() != input_norm_weight_dtype) {
    x_for_norm = x.to(input_norm_weight_dtype);
  }
  x = std::get<0>(input_norm_->forward(x_for_norm));
  // Convert back to original dtype if needed
  if (x.scalar_type() != x_original_dtype) {
    x = x.to(x_original_dtype);
  }

  // Attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);

  // Post-attention norm: x = norm(attn + residual), residual = attn + residual
  // Ensure input dtype matches weight dtype before calling forward
  auto post_norm_weight_dtype = post_norm_->weight().scalar_type();
  auto x_before_post_norm_dtype = x.scalar_type();
  torch::Tensor x_for_post_norm = x;
  if (x.scalar_type() != post_norm_weight_dtype) {
    x_for_post_norm = x.to(post_norm_weight_dtype);
  }
  std::optional<torch::Tensor> residual_for_post_norm = residual;
  if (residual.has_value() &&
      residual.value().scalar_type() != post_norm_weight_dtype) {
    residual_for_post_norm = residual.value().to(post_norm_weight_dtype);
  }
  std::tie(x, residual) =
      post_norm_->forward(x_for_post_norm, residual_for_post_norm);
  // Convert back to original dtype if needed
  if (x.scalar_type() != x_before_post_norm_dtype) {
    x = x.to(x_before_post_norm_dtype);
  }

  // MLP forward; Qwen2 then adds residual (emb + attn) to MLP output.
  // Match diffusers: keep bf16 (no float32).
  x = mlp_->forward(x);
  if (residual.has_value()) {
    x = x + residual.value();
  }
  return x;
}

}  // namespace layer
}  // namespace xllm
