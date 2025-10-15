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

#include "core/framework/model_context.h"
#include "core/layers/qwen3_moe_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class Qwen3MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  Qwen3MoeDecoderLayerImpl(const ModelContext& context, const int32_t i) {
    // register submodules
    decoder_layer_ = register_module("decoder_layer",
                                     layer::Qwen3MoeDecoderLayer(context, i));
  }

#if defined(USE_NPU)
  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor cos_pos,
                        torch::Tensor sin_pos,
                        torch::Tensor attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        torch::Tensor expert_array,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr) {
    return decoder_layer_(x,
                          cos_pos,
                          sin_pos,
                          attn_mask,
                          kv_cache,
                          input_params,
                          expert_array,
                          event,
                          event_flag);
  }
#elif defined(USE_MLU)
  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& positions,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params) {
    return decoder_layer_(x, positions, attn_metadata, kv_cache, input_params);
  }
#endif
  void load_state_dict(const StateDict& state_dict) {
    decoder_layer_->load_state_dict(state_dict);
  }

#if defined(USE_NPU)
  void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights(prefix);
  }

  void merge_loaded_weights() { decoder_layer_->merge_loaded_weights(); }
#endif

 private:
  layer::Qwen3MoeDecoderLayer decoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3MoeDecoderLayer);

torch::Tensor get_qwen3_moe_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
  return get_concat_rotary_embedding(dim, seq_len, rope_theta, options);
}

class Qwen3MoeModelImpl : public torch::nn::Module {
 public:
  Qwen3MoeModelImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    // register submodules
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    num_speculative_tokens_ = model_args.num_speculative_tokens();
#if defined(USE_NPU)
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    atb_pos_emb_ = layer::PosEmbedding(context);
    cos_sin_ =
        get_qwen3_moe_rotary_embedding(128,
                                       model_args.max_position_embeddings(),
                                       model_args.rope_theta(),
                                       options);

    max_seq_len_ = model_args.max_position_embeddings();
    int32_t mask_value = model_args.dtype() == "bfloat16" ? 1 : -9984;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    norm_ = register_module("norm", layer::RmsNorm(context));
    mapping_data_ = parallel_args.mapping_data();
#elif defined(USE_MLU)
    norm_ = register_module(
        "norm",
        layer::RmsNorm(
            model_args.hidden_size(), model_args.rms_norm_eps(), options));
    embed_tokens_ =
        register_module("embed_tokens",
                        layer::WordEmbedding(model_args.vocab_size(),
                                             model_args.hidden_size(),
                                             context.get_parallel_args(),
                                             options));
#endif
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = Qwen3MoeDecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    dp_size_ = parallel_args.dp_size();
    std::vector<int64_t> indices;
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
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
#if defined(USE_NPU)
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({0}).to(torch::kInt32).to(device_);
      }
    }

    auto h = embed_tokens_(tokens, 0);
    int64_t input_length = tokens.size(0);
    torch::Tensor expert_array = torch::arange(
        0,
        input_length * num_experts_per_tok_,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    torch::Tensor attn_mask;
    if (num_speculative_tokens_ == 0 || input_params.global_empty_kv_cache) {
      attn_mask = attn_mask_.get_attn_mask(128, dtype_, device_);
    } else {
      attn_mask = attn_mask_.gen_free_mask(
          num_speculative_tokens_ + 1, dtype_, device_);
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
    return norm_(h, 0);
#elif defined(USE_MLU)
    bool is_prefill = input_params.q_max_seq_len > 1;
    auto attn_metadata =
        layer::AttentionMetadata::build(input_params, is_prefill);
    torch::Tensor h = embed_tokens_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, attn_metadata, kv_caches[i], input_params);
    }
    return norm_(h);
#endif
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.get_dict_with_prefix("embed_tokens."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

#if defined(USE_NPU)
  void verify_loaded_weights(const std::string& prefix) const {
    embed_tokens_->verify_loaded_weights(prefix + "embed_tokens.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() {
    embed_tokens_->merge_loaded_weights();
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  std::vector<layer::WordEmbedding> get_word_embedding() {
    return {embed_tokens_};
  }

  void set_word_embedding(std::vector<layer::WordEmbedding>& word_embedding) {
    embed_tokens_ = word_embedding[0];
  }
#endif

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Qwen3MoeDecoderLayer> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t num_experts_per_tok_;
  int32_t num_speculative_tokens_ = 0;
  at::Device device_;
  torch::Dtype dtype_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RmsNorm norm_{nullptr};
#if defined(USE_NPU)
  layer::AttentionMask attn_mask_;
  torch::Tensor cos_sin_;
  layer::PosEmbedding atb_pos_emb_{nullptr};
#endif
};
TORCH_MODULE(Qwen3MoeModel);

class Qwen3MoeForCausalLMImpl : public torch::nn::Module {
 public:
  Qwen3MoeForCausalLMImpl(const ModelContext& context) {
    model_ = register_module("model", Qwen3MoeModel(context));
#if defined(USE_NPU)
    lm_head_ = register_module("lm_head", layer::LmHead(context));
#elif defined(USE_MLU)
    lm_head_ = register_module(
        "lm_head",
        layer::ColumnParallelLinear(context.get_model_args().hidden_size(),
                                    context.get_model_args().vocab_size(),
                                    /*bias=*/false,
                                    /*gather_output=*/true,
                                    context.get_parallel_args(),
                                    context.get_tensor_options()));
#endif
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const std::vector<torch::Tensor>& tokens,
                        const std::vector<torch::Tensor>& positions,
                        std::vector<KVCache>& kv_caches,
                        const std::vector<ModelInputParams>& input_params) {
    return model_(tokens[0], positions[0], kv_caches, input_params[0]);
  }

  // hidden_states: [num_tokens, hidden_size]
  // seleted_idxes: [num_tokens]
  // returns: [num_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
#if defined(USE_NPU)
    return lm_head_(hidden_states, seleted_idxes, 0);
#elif defined(USE_MLU)
    // select tokens if provided
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h, c10::nullopt);
#endif
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix("model."));
      lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
    }

#if defined(USE_NPU)
    // verify
    model_->verify_loaded_weights("model.");
    lm_head_->verify_loaded_weights("lm_head.");

    model_->merge_loaded_weights();
    lm_head_->merge_loaded_weights();
#endif
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

#if defined(USE_NPU)
  layer::LmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  std::vector<layer::WordEmbedding> get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(std::vector<layer::WordEmbedding>& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }
#endif

 private:
  Qwen3MoeModel model_{nullptr};
#if defined(USE_NPU)
  layer::LmHead lm_head_{nullptr};
#elif defined(USE_MLU)
  layer::ColumnParallelLinear lm_head_{nullptr};
#endif
};
TORCH_MODULE(Qwen3MoeForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3_moe, Qwen3MoeForCausalLM);

// register the model args
// example config:
// https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
// https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json
REGISTER_MODEL_ARGS(qwen3_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_moe");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 6144);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 40960);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 48);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 768);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(num_experts, "num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.001f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});
}  // namespace xllm