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

#include "qwen3_decoder_layer.h"

#include <glog/logging.h>

namespace xllm::hf {

std::shared_ptr<Qwen3DecoderImpl> create_qwen3_decode_layer(
    const Context& context) {
  return std::make_shared<Qwen3DecoderImpl>(context);
}

Qwen3DecoderImpl::Qwen3DecoderImpl(const Context& context) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  // Initialize attention layers
  attention_ =
      xllm::Qwen3Attention(model_args, quant_args, parallel_args, options);

  // Initialize norm layers
  input_norm_ = register_module(
      "input_norm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  post_norm_ = register_module(
      "post_norm",
      RMSNorm(model_args.hidden_size(), model_args.rms_norm_eps(), options));

  // Initialize mlp
  mlp_ = xllm::Qwen3MLP(model_args, quant_args, parallel_args, options);

  dtype_ = c10::typeMetaToScalarType(options.dtype());
  rank_id_ = parallel_args.rank();
}

void Qwen3DecoderImpl::load_state_dict(const StateDict& state_dict) {
  // TODO(mlu): verify weight loading for norm, mlp, attention
  attention_.load_state_dict(state_dict.get_dict_with_prefix("self_attn."));
  input_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("input_layernorm."));
  post_norm_->load_state_dict(
      state_dict.get_dict_with_prefix("post_layernorm."));
  mlp_.load_state_dict(state_dict.get_dict_with_prefix("mlp."));
}

torch::Tensor Qwen3DecoderImpl::forward(
    torch::Tensor& x,
    torch::Tensor& positions,
    torch::Tensor& residual,
    const xllm::mlu::AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // x shape: [num_tokens, hidden_size]
  residual = x;

  // Pre-attention norm
  x = input_norm_->forward(x);

  // Attention
  auto attn_out = attention_.forward(positions, x, attn_metadata, kv_cache);

  // Residual connection after attention
  x = attn_out + residual;

  // Post-attention norm
  residual = x;
  x = post_norm_->forward(x);

  // MLP forward
  auto mlp_out = mlp_.forward(x);

  // Final residual connection
  x = mlp_out + residual;

  return x;
}

Qwen3Decoder::Qwen3Decoder(const Context& context)
    : ModuleHolder(create_qwen3_decode_layer(context)) {}

}  // namespace xllm::hf
