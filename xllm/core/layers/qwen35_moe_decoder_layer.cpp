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

#include "qwen35_moe_decoder_layer.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

Qwen35MoeDecoderLayerImpl::Qwen35MoeDecoderLayerImpl(const ModelContext& context,
                                                     int32_t layer_id)
    : parallel_args_(context.get_parallel_args()) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();

  auto layer_types = model_args.layer_types();
  if (layer_types.empty()) {
    int32_t interval = model_args.full_attention_interval();
    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      layer_types.push_back((i + 1) % interval == 0 ? "full_attention"
                                                     : "linear_attention");
    }
  }

  if (layer_id >= 0 && layer_id < static_cast<int32_t>(layer_types.size())) {
    layer_type_ = layer_types[layer_id];
  } else {
    layer_type_ = "full_attention";
  }

  if (layer_type_ == "linear_attention") {
    linear_attention_ =
        register_module("linear_attn", GatedDeltaNet(context));
  } else {
    full_attention_ = register_module("self_attn", Qwen2Attention(context));
  }

  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  auto mlp_only_layers = model_args.mlp_only_layers();
  bool use_moe = model_args.n_routed_experts() > 0 &&
                 (std::count(mlp_only_layers.begin(), mlp_only_layers.end(),
                             layer_id) == 0) &&
                 (layer_id + 1) % model_args.decoder_sparse_step() == 0;

  if (use_moe) {
    moe_mlp_ = register_module("mlp",
                               FusedMoE(model_args,
                                        FusedMoEArgs{.is_gated = true},
                                        quant_args,
                                        parallel_args_,
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
                                    parallel_args_.tp_group_,
                                    options));
  }
}

void Qwen35MoeDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  if (layer_type_ == "linear_attention") {
    linear_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("linear_attn."));
  } else {
    full_attention_->load_state_dict(
        state_dict.get_dict_with_prefix("self_attn."));
  }
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

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
Qwen35MoeDecoderLayerImpl::apply_norm(RMSNorm& norm,
                                      torch::Tensor& input,
                                      std::optional<torch::Tensor>& residual) {
  if (!residual.has_value()) {
    auto new_residual = input;
    auto output = std::get<0>(norm->forward(input));
    return {output, new_residual};
  }
  return norm->forward(input, residual);
}

torch::Tensor Qwen35MoeDecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  std::tie(x, residual) = apply_norm(input_norm_, x, residual);

  if (layer_type_ == "linear_attention") {
    x = linear_attention_->forward(x, positions, attn_metadata, kv_cache);
  } else {
    x = full_attention_->forward(positions, x, attn_metadata, kv_cache);
  }

  std::tie(x, residual) = apply_norm(post_norm_, x, residual);

  if (moe_mlp_) {
    x = moe_mlp_(x, input_params);
  } else {
    x = mlp_(x);
  }

  return x;
}

}  // namespace layer
}  // namespace xllm
