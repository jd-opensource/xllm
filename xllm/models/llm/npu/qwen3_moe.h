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

#include <unordered_map>

#include "core/framework/model/model_output.h"
#include "core/framework/model/npu_dp_ep_padding.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_qwen3_moe_decoder_layer_impl.h"
#include "llm_model_base.h"

namespace xllm {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class Qwen3MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  Qwen3MoeDecoderLayerImpl(const ModelContext& context,
                           const int32_t layer_id) {
    // register submodules
    decoder_layer_ = register_module(
        "decoder_layer", layer::NpuQwen3MoeDecoderLayer(context, layer_id));
  }

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
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
  layer::NpuQwen3MoeDecoderLayer decoder_layer_{nullptr};
};
TORCH_MODULE(Qwen3MoeDecoderLayer);

class Qwen3MoeModelImpl : public LlmModelImplBase<Qwen3MoeDecoderLayer> {
 public:
  Qwen3MoeModelImpl(const ModelContext& context)
      : LlmModelImplBase<Qwen3MoeDecoderLayer>(context.get_model_args()) {
    auto options = context.get_tensor_options();
    auto model_args = context.get_model_args();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());

    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));

    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        128,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);

    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
    norm_ = register_module("norm", layer::NpuRMSNorm(context));

    for (int32_t i = 0; i < model_args.n_layers(); ++i) {
      auto block = Qwen3MoeDecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    num_experts_per_tok_ = model_args.num_experts_per_tok();
  }

 protected:
  void post_process_rotary_pos_embeddings(
      torch::Tensor& cos_pos,
      torch::Tensor& sin_pos,
      const torch::Tensor& positions) override {
    if (positions.dim() != 2) {
      return;
    }

    auto apply = [this](torch::Tensor x) {
      auto freqs_t = x[0].clone();
      int64_t mrop_length = static_cast<int64_t>(freqs_t.size(-1) / 2);

      for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
        int64_t offset = dim_idx;
        int64_t section_len = mrope_section_[dim_idx];
        int64_t length = section_len * 3;

        auto idx_first_half = torch::arange(offset, length, 3, torch::kLong);
        auto idx_second_half = torch::arange(
            offset + mrop_length, length + mrop_length, 3, torch::kLong);
        auto idx_tensor =
            torch::cat({idx_first_half, idx_second_half}, 0).to(x.device());

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

  torch::Tensor build_attention_mask(
      const ModelInputParams& input_params,
      const torch::Tensor& cos_pos,
      const torch::Tensor& hidden_states) override {
    if (input_params.batch_forward_type.is_decode()) {
      return torch::Tensor();
    }

    max_seq_len_ = FLAGS_enable_chunked_prefill
                       ? std::max(input_params.kv_max_seq_len, max_seq_len_)
                       : 128;

    if (FLAGS_enable_chunked_prefill) {
      auto full_mask = attn_mask_.get_attn_mask(
          max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());

      int batch_size = input_params.q_seq_lens_vec.size();
      if (batch_size > 0) {
        std::vector<torch::Tensor> req_mask_vec;
        req_mask_vec.reserve(batch_size);

        for (int j = 0; j < batch_size; j++) {
          int start =
              input_params.kv_seq_lens_vec[j] - input_params.q_seq_lens_vec[j];
          int end = input_params.kv_seq_lens_vec[j];

          req_mask_vec.emplace_back(full_mask.slice(0, start, end));
        }
        return torch::cat(req_mask_vec, 0);
      }
      return full_mask;
    }

    if (input_params.batch_forward_type.is_prefill()) {
      return attn_mask_.get_attn_mask(
          max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());
    }

    return torch::Tensor();
  }

  void mutate_input_params(ModelInputParams& input_params,
                           const torch::Tensor& tokens,
                           const torch::Tensor& hidden_states) override {
    const int64_t input_length = hidden_states.size(0);
    input_params.expert_array =
        torch::arange(0,
                      input_length * num_experts_per_tok_,
                      torch::TensorOptions()
                          .dtype(torch::kInt32)
                          .device(hidden_states.device()));
  }

  void after_layer_forward(size_t layer_id,
                           torch::Tensor& hidden_states,
                           ModelInputParams& input_params) override {
    const auto& deep_stacks = input_params.deep_stacks;
    if (!deep_stacks.empty() && layer_id < deep_stacks.size()) {
      hidden_states = deepstack_process(
          hidden_states, input_params.visual_pos_masks, deep_stacks[layer_id]);
    }
  }

 private:
  torch::Tensor deepstack_process(torch::Tensor hidden_states,
                                  torch::Tensor visual_pos_masks,
                                  torch::Tensor visual_embeds) {
    visual_pos_masks = visual_pos_masks.to(hidden_states.device());
    auto selected = hidden_states.index({visual_pos_masks});
    auto local_this = selected + visual_embeds;
    hidden_states.index_put_({visual_pos_masks}, local_this);
    return hidden_states;
  }

  int32_t max_seq_len_ = 0;
  int32_t num_experts_per_tok_ = 0;
};
TORCH_MODULE(Qwen3MoeModel);

class Qwen3MoeForCausalLMImpl : public LlmForCausalLMImplBase<Qwen3MoeModel> {
 public:
  Qwen3MoeForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Qwen3MoeModel>(context) {}
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
