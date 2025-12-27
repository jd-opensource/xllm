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

  // DeepSeek MoE only support deep ep all2all
  //  when dp_size > 1 for now
  enable_deep_ep_ = FLAGS_expert_parallel_degree == 2 && is_moe_layer_;
  if (enable_deep_ep_) {
    CHECK(parallel_args_.dp_size() > 1)
        << "DeepSeek MoE only supports deep expert parallel (EP) all2all when "
           "dp_size > 1.";
  }

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
  if (is_moe_layer_) {
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

torch::Tensor DeepseekV2DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  // we only support all2all communcation for decode stage for now.
  bool enable_moe_all2all =
      enable_deep_ep_ && input_params.batch_forward_type.is_decode();
  PaddingInfo pad_info;

  // Pre-attention norm
  residual = x;
  x = std::get<0>(input_norm_->forward(x));

  // Attention
  x = attention_->forward(positions, x, attn_metadata, kv_cache);

  // we apply communcation here to avoid implicit communcation in deepseek
  // attention layer. for dp + ep, we will use reduce scatter here instead of
  // all reduce
  if (enable_moe_all2all) {
    // only rank 0 in tp_group will add the residual value
    if (parallel_args_.tp_group_->rank() == 0) {
      x = x + residual.value();
    }
    // if tp_size > 1, we need to pad the input tensor before reduce scatter
    //  to make sure every rank contain at least one token
    if (parallel_args_.tp_group_->world_size() > 1) {
      auto pad_result = check_and_pad_before_scatter(x, parallel_args_);
      x = pad_result.first;
      pad_info = pad_result.second;
      x = xllm::parallel_state::reduce_scatter(x, parallel_args_.tp_group_);
    }
  } else {
    x = xllm::parallel_state::reduce(x, parallel_args_.tp_group_);
    x = x + residual.value();
  }

  // Post-attention norm
  residual = x;
  x = std::get<0>(post_norm_->forward(x));

  // MLP forward
  if (moe_mlp_) {
    x = moe_mlp_(x, input_params);
  } else {
    x = mlp_(x);
  }

  // add up residual after mlp/moe
  x = x + residual.value();

  if (enable_moe_all2all && parallel_args_.tp_group_->world_size() > 1) {
    // unpadding the output after all gather if tp size > 1
    x = parallel_state::gather(x, parallel_args_.tp_group_, 0);
    x = check_and_unpad_after_gather(x, pad_info);
  }
  residual = std::nullopt;
  return x;
}

}  // namespace layer
}  // namespace xllm
