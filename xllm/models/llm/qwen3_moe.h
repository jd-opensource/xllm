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

#include "core/layers/qwen3_moe_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

class Qwen3MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  Qwen3MoeDecoderLayerImpl(const ModelContext& context, const int32_t i) {
    // register submodules
    decoder_layer_ = register_module("decoder_layer",
                                     layer::Qwen3MoeDecoderLayer(context, i));
  }

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& positions,
                        const layer::AttentionMetadata& attn_metadata,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params) {
    return decoder_layer_(
        x, residual, positions, attn_metadata, kv_cache, input_params);
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
    layers_.reserve(model_args.n_layers());

    if (!mrope_section_.empty()) {
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          128,
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
    }

    // register submodules
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    norm_ = register_module("norm", layer::RMSNorm(context));
    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto layer = Qwen3MoeDecoderLayer(context, i);
      layers_.push_back(layer);
    }

    dp_size_ = parallel_args.dp_size();
    dp_local_tp_size_ = parallel_args.world_size() / dp_size_;
    dp_rank_ = parallel_args.rank() / dp_local_tp_size_;
    rank_ = parallel_args.rank();
  }

  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) {
    auto target_cos_sin = cos_sin_.index({positions});
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    auto apply = [this](torch::Tensor x) {
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
    cos_pos = apply(cos_pos.reshape({positions.size(0), -1, cos_pos.size(1)}));
    sin_pos = apply(sin_pos.reshape({positions.size(0), -1, sin_pos.size(1)}));
    return std::make_pair(cos_pos, sin_pos);
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
    ModelInputParams modified_input_params = input_params;
    if (dp_size_ > 1) {
      if (tokens.sizes() == 0) {
        tokens = torch::tensor({1}).to(torch::kInt32).to(device_);
        positions = torch::tensor({1}).to(torch::kInt32).to(device_);
      }
      auto& dp_token_nums = modified_input_params.dp_global_token_nums;
      std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    }
    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }

    auto deep_stacks = input_params.deep_stacks;
    int deep_stack_size = deep_stacks.size();
    auto attn_metadata = layer::AttentionMetadata::build(modified_input_params);
    if (positions.dim() == 2) {
      std::tie(attn_metadata.mrope_cos, attn_metadata.mrope_sin) =
          apply_mrope(positions);
    }

    std::optional<torch::Tensor> residual;
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h,
                residual,
                positions,
                attn_metadata,
                kv_caches[i],
                modified_input_params);

      if (deep_stack_size && i < deep_stack_size) {
        h = deepstack_process(h, input_params.visual_pos_masks, deep_stacks[i]);
      }
    }
    return std::get<0>(norm_(h, residual));
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

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    embed_tokens_ = word_embedding;
  }
  torch::Tensor get_input_embeddings(torch::Tensor input_ids) {
    return embed_tokens_(input_ids);
  }

 private:
  std::vector<Qwen3MoeDecoderLayer> layers_;
  int32_t dp_rank_;
  int32_t rank_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  torch::Device device_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
  torch::Tensor cos_sin_;
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
    // select tokens if provided
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return lm_head_(h);
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
