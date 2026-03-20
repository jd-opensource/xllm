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

#if defined(USE_NPU)
#include <atb/atb_infer.h>

#include "xllm_atb_layers/core/include/atb_speed/log.h"
#endif

#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_qwen3_decoder_layer_impl.h"
#include "models/llm/npu/llm_model_base.h"
#include "models/model_registry.h"

namespace xllm {

class Qwen3_Omni_MoeTalkerCodePredictorModelImpl
    : public LlmModelImplBase<QWen3DecoderLayer> {
 public:
  Qwen3_Omni_MoeTalkerCodePredictorModelImpl(const ModelContext& context)
      : LlmModelImplBase("qwen3_omni_moe_talker_code_predictor",
                         context.get_model_args()) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    auto dp_local_tp_size =
        parallel_args.world_size() / parallel_args.dp_size();
    dp_rank_ = parallel_args.rank() / dp_local_tp_size;

    for (int i = 0; i < model_args.talker_num_code_groups() - 1; ++i) {
      auto embedding = layer::NpuWordEmbedding(context);
      codec_embeddings_.push_back(embedding);
    }

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        128,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen3DecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
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

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    bool use_deepstack = input_params.deep_stacks.size() > 0;
    std::vector<torch::Tensor> deep_stacks;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = get_input_embeddings(tokens, input_params.generation_steps);
    }
    if (use_deepstack) {
      deep_stacks = input_params.deep_stacks;
    }
    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();

    if (positions.dim() == 2) {  // mrope
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

    torch::Tensor attn_mask;
    if (!input_params.batch_forward_type.is_decode()) {
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
        attn_mask = attn_mask_.get_attn_mask(
            128, cos_pos.dtype().toScalarType(), cos_pos.device());
      }
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event{nullptr};
      std::atomic<bool>* event_flag{nullptr};

      if (input_params.layer_synchronizer != nullptr) {
        event = input_params.layer_synchronizer->get_event(i);
        event_flag = input_params.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      auto& layer = layers_[i];

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);
      if (use_deepstack) {
        if (deep_stacks.size() > 0 && i < deep_stacks.size()) {
          h = deepstack_process(
              h, input_params.visual_pos_masks, deep_stacks[i]);
        }
      }
    }
    auto hidden_states = norm_(h, 0);
    return ModelOutput(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) override {
    for (size_t i = 0; i < codec_embeddings_.size(); ++i) {
      std::string prefix = "codec_embedding." + std::to_string(i) + ".";
      codec_embeddings_[i]->load_state_dict(
          state_dict.get_dict_with_prefix(prefix));
    }
    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const override {
    for (size_t i = 0; i < codec_embeddings_.size(); ++i) {
      std::string embed_prefix =
          prefix + "codec_embedding." + std::to_string(i) + ".";
      codec_embeddings_[i]->verify_loaded_weights(embed_prefix);
    }

    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() override {
    for (size_t i = 0; i < codec_embeddings_.size(); ++i) {
      codec_embeddings_[i]->merge_loaded_weights();
    }

    for (size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

  void set_npu_word_embedding(
      layer::NpuWordEmbedding& npu_word_embedding) override {}

  torch::Tensor get_input_embeddings(torch::Tensor input_ids,
                                     int code_group_idx) {
    int idx = std::min(code_group_idx, (int)codec_embeddings_.size() - 1);
    return codec_embeddings_[idx](input_ids, 0);
  }

 private:
  std::vector<layer::NpuWordEmbedding> codec_embeddings_;
  torch::Tensor viusal_pos_mask_;
};
TORCH_MODULE(Qwen3_Omni_MoeTalkerCodePredictorModel);

class Qwen3_Omni_MoeTalkerCodePredictorForCausalImpl
    : public LlmForCausalLMImplBase<Qwen3_Omni_MoeTalkerCodePredictorModel> {
 public:
  Qwen3_Omni_MoeTalkerCodePredictorForCausalImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<Qwen3_Omni_MoeTalkerCodePredictorModel>(context),
        context_(context) {
    model_args_ = context.get_model_args();
    auto options = context.get_tensor_options();

    lm_head_modules = register_module("lm_head", torch::nn::ModuleList());
    for (int i = 0; i < model_args_.talker_num_code_groups() - 1; ++i) {
      auto lm_head = layer::NpuLmHead(context);
      lm_head_modules->push_back(lm_head);
      lm_heads_.push_back(lm_head);
    }
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    std::string prefix = "talker.code_predictor.";

    for (const auto& state_dict : loader->get_state_dicts()) {
      model_->load_state_dict(
          state_dict->get_dict_with_prefix(prefix + "model."));

      for (size_t i = 0; i < lm_heads_.size(); ++i) {
        std::string lm_head_name =
            prefix + "lm_head." + std::to_string(i) + ".";
        lm_heads_[i]->load_state_dict(
            state_dict->get_dict_with_prefix(lm_head_name));
      }
    }
    model_->verify_loaded_weights(prefix + "model.");
    model_->merge_loaded_weights();
    for (size_t i = 0; i < lm_heads_.size(); ++i) {
      std::string lm_head_name = prefix + "lm_head." + std::to_string(i) + ".";
      lm_heads_[i]->verify_loaded_weights(lm_head_name);
      lm_heads_[i]->merge_loaded_weights();
    }
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }

  virtual void update_expert_weight(int32_t layer_id) { return; }

  layer::NpuLmHead get_npu_lm_head(int code_group_idx = 0) {
    return lm_heads_[code_group_idx];
  }

  layer::NpuWordEmbedding get_npu_word_embedding() { return nullptr; }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    model_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  const ModelContext& context_;
  ModelArgs model_args_;
  torch::nn::ModuleList lm_head_modules = nullptr;
  std::vector<layer::NpuLmHead> lm_heads_;
};
TORCH_MODULE(Qwen3_Omni_MoeTalkerCodePredictorForCausal);

REGISTER_CAUSAL_MODEL(qwen3_omni_moe_talker_code_predictor,
                      Qwen3_Omni_MoeTalkerCodePredictorForCausal);

REGISTER_MODEL_ARGS(qwen3_omni_moe_talker_code_predictor, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_omni_moe_talker_code_predictor");

  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 5);

  LOAD_ARG_OR(
      n_layers, "talker_config.code_predictor_config.num_hidden_layers", 5);
  LOAD_ARG_OR(attention_bias,
              "talker_config.code_predictor_config.attention_bias",
              false);
  LOAD_ARG_OR(attention_dropout,
              "talker_config.code_predictor_config.attention_dropout",
              0.0f);
  LOAD_ARG_OR(
      hidden_act, "talker_config.code_predictor_config.hidden_act", "silu");
  LOAD_ARG_OR(
      hidden_size, "talker_config.code_predictor_config.hidden_size", 1024);
  LOAD_ARG_OR(initializer_range,
              "talker_config.code_predictor_config.initializer_range",
              0.02f);
  LOAD_ARG_OR(intermediate_size,
              "talker_config.code_predictor_config.intermediate_size",
              3072);
  LOAD_ARG_OR(max_position_embeddings,
              "talker_config.code_predictor_config.max_position_embeddings",
              32768);
  LOAD_ARG_OR(
      rms_norm_eps, "talker_config.code_predictor_config.rms_norm_eps", 1e-6);
  LOAD_ARG_OR(
      rope_theta, "talker_config.code_predictor_config.rope_theta", 1000000.0f);
  LOAD_ARG_OR(
      vocab_size, "talker_config.code_predictor_config.vocab_size", 2048);

  LOAD_ARG_OR(head_dim, "talker_config.code_predictor_config.head_dim", 128);
  LOAD_ARG_OR(
      n_kv_heads, "talker_config.code_predictor_config.num_key_value_heads", 8);
  LOAD_ARG_OR(
      n_heads, "talker_config.code_predictor_config.num_attention_heads", 16);

  LOAD_ARG_OR(talker_num_code_groups, "talker_config.num_code_groups", 16);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
