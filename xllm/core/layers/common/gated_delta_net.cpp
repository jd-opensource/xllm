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

#include "gated_delta_net.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

GatedDeltaNetImpl::GatedDeltaNetImpl(const ModelContext& context) {
  const auto& model_args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();

  tp_size_ = parallel_args.tp_size();
  tp_rank_ = parallel_args.tp_rank();

  hidden_size_ = model_args.hidden_size();
  num_k_heads_ = model_args.linear_num_key_heads();
  num_v_heads_ = model_args.linear_num_value_heads();
  head_k_dim_ = model_args.linear_key_head_dim();
  head_v_dim_ = model_args.linear_value_head_dim();
  key_dim_ = head_k_dim_ * num_k_heads_;
  value_dim_ = head_v_dim_ * num_v_heads_;
  conv_kernel_size_ = model_args.linear_conv_kernel_dim();
  layer_norm_epsilon_ = model_args.rms_norm_eps();

  int64_t conv_dim = key_dim_ * 2 + value_dim_;

  conv1d_ = register_module(
      "conv1d",
      ColumnParallelLinear(conv_kernel_size_, conv_dim, false, true, quant_args,
                           parallel_args.tp_group_, options));

  in_proj_qkvz_ = register_module(
      "in_proj_qkvz",
      ColumnParallelLinear(hidden_size_, key_dim_ * 2 + value_dim_ * 2, false,
                           true, quant_args, parallel_args.tp_group_, options));

  in_proj_ba_ = register_module(
      "in_proj_ba",
      ColumnParallelLinear(hidden_size_, num_v_heads_ * 2, false, true,
                           quant_args, parallel_args.tp_group_, options));

  norm_ = register_module(
      "norm", RMSNorm(head_v_dim_, layer_norm_epsilon_, options));

  out_proj_ = register_module(
      "out_proj",
      RowParallelLinear(value_dim_, hidden_size_, false, true, true, quant_args,
                        parallel_args.tp_group_, options));

  A_log_ = register_parameter(
      "A_log",
      torch::empty({num_v_heads_ / tp_size_}, options.dtype(torch::kFloat32)));

  dt_bias_ = register_parameter(
      "dt_bias", torch::ones({num_v_heads_ / tp_size_}, options));
}

void GatedDeltaNetImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.contains("conv1d.weight")) {
    auto conv_weight = state_dict.get_tensor("conv1d.weight");
    conv_weight = conv_weight.unsqueeze(1);
    conv1d_->load_state_dict(state_dict.get_dict_with_prefix("conv1d."));
  }

  in_proj_qkvz_->load_state_dict(
      state_dict.get_dict_with_prefix("in_proj_qkvz."));
  in_proj_ba_->load_state_dict(state_dict.get_dict_with_prefix("in_proj_ba."));
  out_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));

  if (state_dict.contains("A_log")) {
    A_log_.copy_(state_dict.get_tensor("A_log"));
  }
  if (state_dict.contains("dt_bias")) {
    dt_bias_.copy_(state_dict.get_tensor("dt_bias"));
  }
}

torch::Tensor GatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  int64_t num_tokens = hidden_states.size(0);

  auto mixed_qkvz = in_proj_qkvz_(hidden_states);
  int64_t qkv_size = (key_dim_ * 2 + value_dim_) / tp_size_;
  int64_t z_size = value_dim_ / tp_size_;

  auto chunks = mixed_qkvz.split({qkv_size, z_size}, -1);
  auto mixed_qkv = chunks[0];
  auto z = chunks[1];

  z = z.reshape({z.size(0), -1, head_v_dim_});

  auto ba = in_proj_ba_(hidden_states);
  auto ba_chunks = ba.chunk(2, -1);
  auto b = ba_chunks[0].contiguous();
  auto a = ba_chunks[1].contiguous();

  auto core_attn_out = torch::zeros(
      {num_tokens, num_v_heads_ / tp_size_, head_v_dim_},
      hidden_states.options());

  auto z_shape = z.sizes();
  core_attn_out = core_attn_out.reshape({-1, core_attn_out.size(-1)});
  z = z.reshape({-1, z.size(-1)});

  core_attn_out = std::get<0>(norm_(core_attn_out));
  core_attn_out = core_attn_out.reshape(z_shape);

  core_attn_out = core_attn_out.reshape({num_tokens, -1});
  auto output = out_proj_(core_attn_out);

  return output;
}

}  // namespace layer
}  // namespace xllm
