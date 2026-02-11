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

#pragma once

#include "core/framework/model/model_output.h"
#include "core/framework/model/npu_dp_ep_padding.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_glm4_moe_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class Glm4MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  Glm4MoeDecoderLayerImpl(const ModelContext& context, const int32_t i) {
    // register submodules
    decoder_layer_ =
        register_module("decoder_layer", layer::NpuGlm4MoeDecoder(context, i));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor cos_pos,
                        torch::Tensor sin_pos,
                        torch::Tensor attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        aclrtEvent* event,
                        std::atomic<bool>* event_flag) {
    return decoder_layer_(x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          event,
                          event_flag);
  }

  void load_state_dict(const StateDict& state_dict) {
    decoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights(prefix);
  }

  void merge_loaded_weights() { decoder_layer_->merge_loaded_weights(); }

 private:
  layer::NpuGlm4MoeDecoder decoder_layer_{nullptr};
};
TORCH_MODULE(Glm4MoeDecoderLayer);

class Glm4MoeModelImpl : public LlmModelImplBase<Glm4MoeDecoderLayer> {
 public:
  Glm4MoeModelImpl(const ModelContext& context)
      : LlmModelImplBase<Glm4MoeDecoderLayer>(context.get_model_args()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    num_speculative_tokens_ = model_args.num_speculative_tokens();
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));

    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        64,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
    mrope_section_ = model_args.rope_scaling_mrope_section();

    // int32_t mask_value = model_args.dtype() == "bfloat16" ? 1 : -9984;
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = Glm4MoeDecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    num_experts_per_tok_ = model_args.num_experts_per_tok();
  }

 protected:
  void post_process_rotary_pos_embeddings(
      torch::Tensor& cos_pos,
      torch::Tensor& sin_pos,
      const torch::Tensor& positions) override {
    LlmModelImplBase<Glm4MoeDecoderLayer>::post_process_rotary_pos_embeddings(
        cos_pos, sin_pos, positions);
    cos_pos = cos_pos.view(at::IntArrayRef{-1, 2, cos_pos.size(-1) / 2});
    sin_pos = sin_pos.view(at::IntArrayRef{-1, 2, sin_pos.size(-1) / 2});
  }

  torch::Tensor build_attention_mask(
      const ModelInputParams& input_params,
      const torch::Tensor& cos_pos,
      const torch::Tensor& hidden_states) override {
    if (FLAGS_enable_chunked_prefill) {
      int max_kv_seq = input_params.kv_max_seq_len;
      int num_sequences = input_params.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);

        for (int j = 0; j < num_sequences; j++) {
          auto mask =
              attn_mask_.gen_append_mask(input_params.q_seq_lens_vec[j],
                                         input_params.kv_seq_lens_vec[j],
                                         max_kv_seq,
                                         cos_pos.dtype().toScalarType(),
                                         cos_pos.device());
          req_mask_vec.emplace_back(mask);
        }
        return torch::cat(req_mask_vec, 0);
      }
    }

    if (input_params.batch_forward_type.is_prefill()) {
      return attn_mask_.get_attn_mask(
          128, hidden_states.dtype().toScalarType(), hidden_states.device());
    }

    return torch::Tensor();
  }

  void mutate_input_params(ModelInputParams& input_params,
                           const torch::Tensor& tokens,
                           const torch::Tensor& hidden_states) override {
    int64_t input_length = tokens.size(0);
    input_params.expert_array = torch::arange(
        0,
        input_length * num_experts_per_tok_,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
  }

 private:
  int32_t num_experts_per_tok_ = 0;
  int32_t num_speculative_tokens_ = 0;
};
TORCH_MODULE(Glm4MoeModel);

class Glm4MoeForCausalLMImpl : public LlmForCausalLMImplBase<Glm4MoeModel> {
 public:
  Glm4MoeForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Glm4MoeModel>(context) {}
};
TORCH_MODULE(Glm4MoeForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(glm4_moe, Glm4MoeForCausalLM);

// register the model args
// example config:
// https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json
REGISTER_MODEL_ARGS(glm4_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4_moe");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id_vec, "eos_token_id", std::vector<int>{151329});
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6144);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 40960);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 1536);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_shared_experts, "n_shared_experts", 1);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 96);
  LOAD_ARG_OR(num_experts, "n_routed_experts", 160);
  LOAD_ARG_OR(n_group, "n_group", 1);
  LOAD_ARG_OR(topk_group, "topk_group", 1);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151552);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);

  SET_ARG(stop_token_ids,
          std::unordered_set<int32_t>(args->eos_token_id_vec().begin(),
                                      args->eos_token_id_vec().end()));
});
}  // namespace xllm
