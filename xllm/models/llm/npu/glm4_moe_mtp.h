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

#include <boost/algorithm/string.hpp>

#include "core/framework/model/npu_dp_ep_padding.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_column_parallel_linear_impl.h"
#include "core/layers/npu/npu_glm4_moe_decoder_layer.h"
#include "glm4_moe.h"
#include "llm_model_base.h"

namespace xllm::hf {

class Glm4MoeMtpModelImpl : public torch::nn::Module {
 public:
  Glm4MoeMtpModelImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();
    auto options = context.get_tensor_options();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    // register submodules
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    num_speculative_tokens_ = model_args.num_speculative_tokens();
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    atb_pos_emb_ = layer::PosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        64,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);

    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = Glm4MoeDecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    eh_proj_ = register_module("eh_proj", layer::ColumnParallelLinear(context));
    enorm_ = register_module("enorm", layer::RMSNorm(context));
    hnorm_ = register_module("hnorm", layer::RMSNorm(context));
    final_norm_ = register_module("final_norm", layer::RMSNorm(context));

    dp_size_ = parallel_args.dp_size();
    std::vector<int64_t> indices;
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
    mapping_data_ = parallel_args.mapping_data();
    num_experts_per_tok_ = model_args.num_experts_per_tok();
    for (int i = 0; i < parallel_args.world_size(); i += dp_local_tp_size_) {
      indices.push_back(i);
    }
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({0}).to(torch::kInt32).to(device_);
      }
    }

    torch::Tensor h = embed_tokens_(tokens, 0);
    torch::Tensor enorm = enorm_(h, 0);
    torch::Tensor input_embedding = input_params.input_embedding;
    if (input_embedding.defined()) {
      h = input_embedding;
    } else {
      LOG(WARNING) << "hnorm use embedding from tokens.";
    }

    torch::Tensor hnorm = hnorm_(h, 0);
    CHECK_EQ(enorm.dim(), hnorm.dim());
    CHECK_EQ(enorm.size(0), hnorm.size(0));
    h = torch::cat({enorm, hnorm}, /*dim=*/-1);
    h = eh_proj_(h, 0);

    auto cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto cos_sin_chunks = cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = cos_sin_chunks[0].contiguous();
    auto sin_pos = cos_sin_chunks[1].contiguous();
    cos_pos = cos_pos.view(at::IntArrayRef{-1, 2, cos_pos.size(-1) / 2});
    sin_pos = sin_pos.view(at::IntArrayRef{-1, 2, sin_pos.size(-1) / 2});

    torch::Tensor attn_mask;
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
        attn_mask = torch::cat(req_mask_vec, 0);
      }
    } else {
      if (num_speculative_tokens_ == 0 || input_params.global_empty_kv_cache) {
        attn_mask = attn_mask_.get_attn_mask(128, dtype_, device_);
      } else {
        attn_mask = attn_mask_.gen_free_mask(
            num_speculative_tokens_ + 1, dtype_, device_);
      }
    }

    int64_t input_length = tokens.size(0);
    torch::Tensor expert_array = torch::arange(
        0,
        input_length * num_experts_per_tok_,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));

    // TODO(liangzhiwei20): MTP need more support for layer wise copy.
    if (input_params.layer_wise_load_synchronizer != nullptr) {
      LOG(FATAL) << "MTP not support layer wise copy!";
    }

    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.layer_synchronizer != nullptr) {
        event = input_params.layer_synchronizer->get_event(i);
        event_flag = input_params.layer_synchronizer->get_event_flag(i);
      }

      auto& layer = layers_[i];
      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params,
            expert_array,
            event,
            event_flag);
    }
    return final_norm_(h, 0);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // embed_tokens_->load_state_dict(state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    eh_proj_->load_state_dict(state_dict.get_dict_with_prefix("eh_proj."));
    enorm_->load_state_dict(state_dict.get_dict_with_prefix("enorm."));
    hnorm_->load_state_dict(state_dict.get_dict_with_prefix("hnorm."));
    final_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("shared_head.norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    // embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    eh_proj_->verify_loaded_weights(prefix + "eh_proj.");
    enorm_->verify_loaded_weights(prefix + "enorm.");
    hnorm_->verify_loaded_weights(prefix + "hnorm.");
    final_norm_->verify_loaded_weights(prefix + "shared_head.norm.");
  }

  void merge_loaded_weights() {
    // embed_tokens_->merge_loaded_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    eh_proj_->merge_loaded_weights();
    enorm_->merge_loaded_weights();
    hnorm_->merge_loaded_weights();
    final_norm_->merge_loaded_weights();
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Glm4MoeDecoderLayer> layers_;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  nlohmann::json mapping_data_;
  int32_t num_experts_per_tok_;
  int32_t num_speculative_tokens_ = 0;
  at::Device device_;
  torch::Dtype dtype_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::AttentionMask attn_mask_;
  torch::Tensor cos_sin_;
  layer::PosEmbedding atb_pos_emb_{nullptr};
  layer::ColumnParallelLinear eh_proj_{nullptr};
  layer::RMSNorm enorm_{nullptr};
  layer::RMSNorm hnorm_{nullptr};
  layer::RMSNorm final_norm_{nullptr};
};
TORCH_MODULE(Glm4MoeMtpModel);

class Glm4MoeMtpForCausalLMImpl : public torch::nn::Module {
 public:
  Glm4MoeMtpForCausalLMImpl(const ModelContext& context) {
    model_ = register_module("model", Glm4MoeMtpModel(context));
    // lm_head_ = register_module(
    //     "lm_head", LlmHead(context));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    // select tokens if provided
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  // load model
  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix("model."));
      // lm_head_->load_state_dict(state_dict.get_dict_with_prefix("model.shared_head.head."));
    }

    // verify
    model_->verify_loaded_weights("model.");
    // lm_head_->verify_loaded_weights("model.shared_head.head.");

    model_->merge_loaded_weights();
    // lm_head_->merge_loaded_weights();
  }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    return;
  }
  void update_expert_weight(int32_t layer_id) { return; }
  layer::LmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 private:
  Glm4MoeMtpModel model_{nullptr};
  layer::LmHead lm_head_{nullptr};
};
TORCH_MODULE(Glm4MoeMtpForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(glm4_moe_mtp, Glm4MoeMtpForCausalLM);

// example config:
// https://huggingface.co/zai-org/GLM-4.5-Air/blob/main/config.json
REGISTER_MODEL_ARGS(glm4_moe_mtp, [&] {
  LOAD_ARG_OR(model_type, "model_type", "glm4_moe_mtp");
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
}  // namespace xllm::hf
