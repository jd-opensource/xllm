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
#include <filesystem>
#include <vector>

#include "core/framework/model_context.h"
#include "core/layers/common/layer_utils.h"
#include "core/layers/qwen3_next_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class Qwen3NextModelImpl : public torch::nn::Module {
 public:
  Qwen3NextModelImpl(const ModelContext& context)
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
    norm_ = register_module(
        "norm",
        xllm::layer::Qwen3NextRMSNorm(
            model_args.hidden_size(), model_args.rms_norm_eps(), options));
#if defined(USE_NPU_TORCH)
  embed_tokens_  = layer::WordEmbedding(model_args.vocab_size(),
                                               model_args.hidden_size(),
                                               context.get_parallel_args(),
                                               options);
#else
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      npu_embed_tokens_.push_back(layer::NpuWordEmbedding(context));
    }
#endif
#endif
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = Qwen3NextDecoderLayer(context, i);
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
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({0}).to(torch::kInt32).to(device_);
      }
    }

#if defined(USE_NPU) && defined(USE_NPU_TORCH)
    // Create attention mask
    torch::Tensor attn_mask;
    max_seq_len_ = std::max(input_params.kv_max_seq_len, max_seq_len_);
    
    if (FLAGS_enable_chunked_prefill) {
      int num_sequences = input_params.num_sequences;
      if (num_sequences > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(num_sequences);

        for (int j = 0; j < num_sequences; j++) {
          auto mask = attn_mask_.gen_append_mask(
              input_params.q_seq_lens_vec[j],
              input_params.kv_seq_lens_vec[j],
              max_seq_len_,
              dtype_,
              device_);
          req_mask_vec.emplace_back(mask);
        }
        attn_mask = torch::cat(req_mask_vec, 0);
      }
    } else {
      attn_mask = attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
    }

    layer::AttentionMetadata attn_metadata =
        layer::AttentionMetadata::build(input_params, input_params.q_max_seq_len > 1, attn_mask);
    torch::Tensor h = embed_tokens_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, attn_metadata, kv_caches[i], input_params);
    }
    h = norm_(h);
    return h;
#endif
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
      state_dict.get_dict_with_prefix("embed_tokens."));
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  std::vector<layer::NpuWordEmbedding> get_word_embedding() {
    return {npu_embed_tokens_};
  }

  void set_word_embedding(
      std::vector<layer::NpuWordEmbedding>& word_embedding) {
    npu_embed_tokens_ = word_embedding;
  }


 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Qwen3NextDecoderLayer> layers_;
  int32_t max_seq_len_ = 0;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  nlohmann::json mapping_data_;
  int32_t num_experts_per_tok_;
  int32_t num_speculative_tokens_ = 0;
  at::Device device_;
  torch::Dtype dtype_;
  std::vector<layer::NpuWordEmbedding> npu_embed_tokens_;
  layer::Qwen3NextRMSNorm norm_{nullptr};
  layer::AttentionMask attn_mask_;

#if defined(USE_NPU) && defined(USE_NPU_TORCH)
  layer::WordEmbedding embed_tokens_{nullptr};
#endif
};

TORCH_MODULE(Qwen3NextModel);

class Qwen3NextForCausalLMImpl : public torch::nn::Module {
 public:
  Qwen3NextForCausalLMImpl(const ModelContext& context) {
    model_ = register_module("model", Qwen3NextModel(context));
#if defined(USE_NPU) && defined(USE_NPU_TORCH)
    lm_head_ =
        register_module("lm_head",
                        layer::LmHead(context.get_model_args().hidden_size(),
                                      context.get_model_args().vocab_size(),
                                      /*bias=*/false,
                                      /*gather_output=*/true,
                                      QuantArgs{},
                                      context.get_parallel_args(),
                                      context.get_tensor_options()));
#else
    npu_lm_head_ = register_module("lm_head", layer::NpuLmHead(context));
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
#if defined(USE_NPU) && defined(USE_NPU_TORCH)
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
#endif
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix("model."));
#if defined(USE_NPU_TORCH)
      lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
#else
      npu_lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
#endif
    }

#if defined(USE_NPU) && !defined(USE_NPU_TORCH)
    // verify
    model_->verify_loaded_weights("model.");
    npu_lm_head_->verify_loaded_weights("lm_head.");

    model_->merge_loaded_weights();
    npu_lm_head_->merge_loaded_weights();
#endif
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

#if defined(USE_NPU)

  layer::NpuLmHead get_lm_head() { return npu_lm_head_; }

  void set_lm_head(layer::NpuLmHead& head) { npu_lm_head_ = head; }

  std::vector<layer::NpuWordEmbedding> get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(
      std::vector<layer::NpuWordEmbedding>& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }
#endif

 private:
  layer::NpuLmHead npu_lm_head_{nullptr};
  layer::LmHead lm_head_{nullptr};
  Qwen3NextModel model_{nullptr};

};
TORCH_MODULE(Qwen3NextForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3_next, Qwen3NextForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(qwen3_next, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_next");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(head_dim, "head_dim", 256);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 5120);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 262144);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 512);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(num_experts, "num_experts", 512);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 10);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 48);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 2);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000000.0f);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.001f);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  // Additional parameters for Qwen3-Next architecture
  LOAD_ARG_OR(attn_output_gate, "attn_output_gate", true);
  LOAD_ARG_OR(full_attention_interval, "full_attention_interval", 4);
  LOAD_ARG_OR(linear_conv_kernel_dim, "linear_conv_kernel_dim", 4);
  LOAD_ARG_OR(linear_key_head_dim, "linear_key_head_dim", 128);
  LOAD_ARG_OR(linear_num_key_heads, "linear_num_key_heads", 16);
  LOAD_ARG_OR(linear_num_value_heads, "linear_num_value_heads", 32);
  LOAD_ARG_OR(linear_value_head_dim, "linear_value_head_dim", 128);
  LOAD_ARG_OR(partial_rotary_factor, "partial_rotary_factor", 0.25f);
  LOAD_ARG_OR(shared_expert_intermediate_size, "shared_expert_intermediate_size", 512);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm

