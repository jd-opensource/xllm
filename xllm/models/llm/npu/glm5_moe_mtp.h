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

#pragma once

#include "core/layers/common/rotary_embedding_util.h"
#include "deepseek_v32.h"
#include "mtp_model_base.h"

namespace xllm {

class Glm5MoeMtpModelImpl : public MtpModelImplBase<DeepseekV32DecoderLayer> {
 public:
  Glm5MoeMtpModelImpl(const ModelContext& context)
      : MtpModelImplBase<DeepseekV32DecoderLayer>("glm5_moe_mtp", context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    int32_t mask_value = model_args.dtype() == "bfloat16" ? 1 : -9984;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        model_args.qk_rope_head_dim(),
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
  }
};
TORCH_MODULE(Glm5MoeMtpModel);

class Glm5MoeMtpForCausalLMImpl
    : public MtpForCausalLMImplBase<Glm5MoeMtpModel> {
 public:
  Glm5MoeMtpForCausalLMImpl(const ModelContext& context)
      : MtpForCausalLMImplBase<Glm5MoeMtpModel>(context) {}
};
TORCH_MODULE(Glm5MoeMtpForCausalLM);

using GlmMoeDsaMtpForCausalLM = Glm5MoeMtpForCausalLM;

// register the causal model
REGISTER_CAUSAL_MODEL(glm5_moe_mtp, Glm5MoeMtpForCausalLM);
REGISTER_CAUSAL_MODEL(glm_moe_dsa_mtp, GlmMoeDsaMtpForCausalLM);

REGISTER_MODEL_ARGS_LOADER(glm_moe_dsa_mtp,
                           [](const JsonReader& json, ModelArgs* args) {
                             auto loader = ModelRegistry::get_model_args_loader(
                                 "glm5_moe_mtp");
                             return loader != nullptr && loader(json, args);
                           });

REGISTER_MODEL_ARGS(glm5_moe_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm5_moe_mtp");
  LOAD_ARG_OR(dtype, "dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);

  LOAD_ARG_OR(vocab_size, "vocab_size", 154880);
  LOAD_ARG_OR(hidden_size, "hidden_size", 6144);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 64);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 12288);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 202752);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  // LOAD_ARG_OR(eos_token_id, "eos_token_id", 1);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int>{154820});

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 61);

  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 3);
  LOAD_ARG_OR(moe_layer_freq, "moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "n_routed_experts", 256);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 2048);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 192);
  LOAD_ARG_OR(qk_rope_head_dim, "qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "v_head_dim", 256);
  LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 2048);
  LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 512);
  LOAD_ARG_OR(index_head_dim, "index_head_dim", 128);
  LOAD_ARG_OR(index_n_heads, "index_n_heads", 0);
  LOAD_ARG_OR(index_topk, "index_topk", 2048);

  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return 256;  // args->qk_nope_head_dim() + args->qk_rope_head_dim();
  });
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});
}  // namespace xllm
