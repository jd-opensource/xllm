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

#include <glog/logging.h>

#include <boost/algorithm/string.hpp>

#include "core/framework/model/npu_dp_ep_padding.h"
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

  void load_state_dict(const StateDict& state_dict) {
    auto experts_state_dict = state_dict.get_dict_with_prefix("mlp.experts.");
    auto fused_gate_up = experts_state_dict.get_tensor("gate_up_proj");
    auto fused_down = experts_state_dict.get_tensor("down_proj");

    bool is_fused = fused_gate_up.defined() && fused_down.defined();

    if (is_fused) {
      torch::Tensor expert_gate_up = fused_gate_up;
      torch::Tensor expert_down = fused_down;

      const int num_experts = expert_gate_up.size(0);

      auto chunks = expert_gate_up.chunk(2, /*dim=*/-1);
      auto expert_gate = chunks[0].contiguous();
      auto expert_up = chunks[1].contiguous();

      std::unordered_map<std::string, torch::Tensor> out_state_dict;
      for (const auto& [name, tensor] : state_dict) {
        if (name.find("self_attn.") == 0 || name.find("mlp.gate.") == 0 ||
            name.find("input_layernorm.") == 0 ||
            name.find("post_attention_layernorm.") == 0) {
          out_state_dict.emplace(name, tensor);
        }
      }

      for (int i = 0; i < num_experts; ++i) {
        auto gate_i = expert_gate[i].transpose(0, 1);
        auto up_i = expert_up[i].transpose(0, 1);
        auto down_i = expert_down[i].transpose(0, 1);

        const std::string base = "mlp.experts." + std::to_string(i) + ".";
        out_state_dict.emplace(base + "gate_proj.weight", gate_i);
        out_state_dict.emplace(base + "up_proj.weight", up_i);
        out_state_dict.emplace(base + "down_proj.weight", down_i);
      }
      decoder_layer_->load_state_dict(StateDict(std::move(out_state_dict)));
    } else {
      decoder_layer_->load_state_dict(state_dict);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    decoder_layer_->verify_loaded_weights(prefix);
  }

  void merge_loaded_weights() { decoder_layer_->merge_loaded_weights(); }

 private:
  layer::Qwen3MoeDecoderLayer decoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3MoeDecoderLayer);

class Qwen3MoeModelImpl : public torch::nn::Module {
 public:
  Qwen3MoeModelImpl(const ModelContext& context)
      : device_(context.get_tensor_options().device()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();
    auto parallel_args = context.get_parallel_args();
    mrope_section_ = model_args.rope_scaling_mrope_section();
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    // register submodules
    device_ = options.device();
    dtype_ = options.dtype().toScalarType();
    num_speculative_tokens_ = model_args.num_speculative_tokens();
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));

    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        128,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);

    atb_pos_emb_ = layer::PosEmbedding(context);
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    norm_ = register_module("norm", layer::RMSNorm(context));
    mapping_data_ = parallel_args.mapping_data();

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

  torch::Tensor deepstack_process(torch::Tensor hidden_states,
                                  torch::Tensor visual_pos_masks,
                                  torch::Tensor visual_embeds) {
    visual_pos_masks = visual_pos_masks.to(hidden_states.device());
    auto selected = hidden_states.index({visual_pos_masks});
    auto local_this = selected + visual_embeds;
    hidden_states.index_put_({visual_pos_masks}, local_this);
    return hidden_states;
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
    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens, 0);
    }

    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    if (positions.dim() == 2) {  // mrope
      auto apply = [this](torch::Tensor x) {
        // auto sections = mrope_section_;
        auto freqs_t = x[0].clone();
        for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
          int64_t offset = dim_idx;  // H -> offset=1, W -> offset=2
          int64_t section_len = mrope_section_[dim_idx];
          int64_t length = section_len * 3;

          // indices: [offset, offset+3, offset+6, ..., < length]
          auto idx_first_half = torch::arange(offset, length, 3, torch::kLong);
          auto idx_second_half = torch::arange(offset, length, 3, torch::kLong);
          auto idx_tensor =
              torch::cat({idx_first_half, idx_second_half}, 0).to(x.device());
          // freqs_t[..., idx] = freqs[dim_idx][..., idx]
          auto src = x[dim_idx].index_select(-1, idx_tensor);
          freqs_t.index_copy_(-1, idx_tensor, src);
        }
        return freqs_t;
      };
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }

    torch::Tensor attn_mask;
    max_seq_len_ = FLAGS_enable_chunked_prefill
                       ? std::max(input_params.kv_max_seq_len, max_seq_len_)
                       : 128;
    if (FLAGS_enable_chunked_prefill) {
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());

      int batch_size = input_params.q_seq_lens_vec.size();
      if (batch_size > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(batch_size);

        for (int j = 0; j < batch_size; j++) {
          int start =
              input_params.kv_seq_lens_vec[j] - input_params.q_seq_lens_vec[j];
          int end = input_params.kv_seq_lens_vec[j];

          auto req_mask_slice = attn_mask.slice(0, start, end);
          req_mask_vec.emplace_back(req_mask_slice);
        }
        attn_mask = torch::cat(req_mask_vec, 0);
      }
    } else if (input_params.global_empty_kv_cache) {
      attn_mask = attn_mask_.get_attn_mask(max_seq_len_, dtype_, device_);
    }
    auto deep_stacks = input_params.deep_stacks;
    int deep_stack_size = deep_stacks.size();

    int64_t input_length = h.size(0);
    torch::Tensor expert_array = torch::arange(
        0,
        input_length * num_experts_per_tok_,
        torch::TensorOptions().dtype(torch::kInt32).device(tokens.device()));
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event = nullptr;
      std::atomic<bool>* event_flag = nullptr;
      if (input_params.layer_synchronizer != nullptr) {
        event = input_params.layer_synchronizer->get_event(i);
        event_flag = input_params.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return torch::Tensor();
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
      if (deep_stack_size && i < deep_stack_size) {
        h = deepstack_process(h, input_params.visual_pos_masks, deep_stacks[i]);
      }
    }
    return norm_(h, 0);
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

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }
  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids, 0);
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<Qwen3MoeDecoderLayer> layers_;
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
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::AttentionMask attn_mask_;
  layer::RMSNorm norm_{nullptr};
  torch::Tensor cos_sin_;
  layer::PosEmbedding atb_pos_emb_{nullptr};
  std::vector<int64_t> mrope_section_;
};
TORCH_MODULE(Qwen3MoeModel);

class Qwen3MoeForCausalLMImpl : public torch::nn::Module {
 public:
  Qwen3MoeForCausalLMImpl(const ModelContext& context) {
    model_ = register_module("model", Qwen3MoeModel(context));
    lm_head_ = register_module("lm_head", layer::LmHead(context));
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
    return lm_head_(hidden_states, seleted_idxes, 0);
  }

  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return model_->get_input_embeddings(input_ids);
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model." /*llm model weight prefix*/) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(state_dict->get_dict_with_prefix(prefix));
      lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
    }

    // verify
    model_->verify_loaded_weights(prefix);
    lm_head_->verify_loaded_weights("lm_head.");

    model_->merge_loaded_weights();
    lm_head_->merge_loaded_weights();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  layer::LmHead get_lm_head() { return lm_head_; }

  void set_lm_head(layer::LmHead& head) { lm_head_ = head; }

  layer::WordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    model_->set_word_embedding(word_embedding);
  }

 private:
  Qwen3MoeModel model_{nullptr};
  layer::LmHead lm_head_{nullptr};
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
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(vocab_size, "vocab_size", 151936);
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", std::vector<int>());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});
}  // namespace xllm
