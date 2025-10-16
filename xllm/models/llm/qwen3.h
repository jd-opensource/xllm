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

#include "core/layers/qwen3_decoder_layer.h"
#include "llm_model_base.h"

namespace xllm {

class QWen3DecoderLayerImpl
    : public LlmDecoderLayerImplBase<layer::Qwen3DecoderLayer> {
 public:
  QWen3DecoderLayerImpl(const ModelContext& context)
      : LlmDecoderLayerImplBase<layer::Qwen3DecoderLayer>(context) {}
};
TORCH_MODULE(QWen3DecoderLayer);

class QWen3ModelImpl : public LlmModelImplBase<QWen3DecoderLayer> {
 public:
  QWen3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<QWen3DecoderLayer>("qwen3", context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
#if defined(USE_NPU)
    norm_ = register_module("norm", layer::RmsNorm(context));
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      embed_tokens_.push_back(layer::WordEmbedding(context));
      atb_pos_embeds_.push_back(layer::PosEmbedding(context));
    }
    cos_sin_ = get_concat_rotary_embedding(128,
                                           model_args.max_position_embeddings(),
                                           model_args.rope_theta(),
                                           options);
    int32_t mask_value = FLAGS_enable_chunked_prefill ? -9984 : 1;
    // encode_attn_mask_ =
    //   layer::AttentionMask(options.device(),
    //   options.dtype()).get_attn_mask(2048, options.device(),
    //   options.dtype());
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);
#else
    norm_ = register_module(
        "norm",
        layer::RmsNorm(
            model_args.hidden_size(), model_args.rms_norm_eps(), options));
    for (auto i = 0; i < FLAGS_micro_batch_num; i++) {
      embed_tokens_.push_back(layer::WordEmbedding(model_args.vocab_size(),
                                                   model_args.hidden_size(),
                                                   context.get_parallel_args(),
                                                   options));
    }
#endif

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen3DecoderLayer(context);
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

  virtual torch::Tensor forward(
      std::vector<torch::Tensor> tokens,
      std::vector<torch::Tensor> positions,
      std::vector<KVCache>& kv_caches,
      const std::vector<ModelInputParams>& input_params) {
    auto micro_batch_num = tokens.size();
    std::vector<torch::Tensor> hs;
    hs.reserve(micro_batch_num);
    std::vector<std::vector<torch::Tensor>> deep_stacks;
    deep_stacks.reserve(micro_batch_num);
    bool use_deepstack = input_params[0].deep_stacks.size() > 0;
    std::vector<torch::Tensor> cos_poss;
    cos_poss.reserve(micro_batch_num);
    std::vector<torch::Tensor> sin_poss;
    sin_poss.reserve(micro_batch_num);
    std::vector<torch::Tensor> attn_masks;
    attn_masks.reserve(micro_batch_num);
    std::vector<ModelInputParams>& input_params_news =
        const_cast<std::vector<ModelInputParams>&>(input_params);

    for (auto i = 0; i < micro_batch_num; ++i) {
      if (tokens[i].numel() == 0) {
        tokens[i] = torch::tensor({1}).to(torch::kInt32).to(tokens[0].device());
        positions[i] =
            torch::tensor({0}).to(torch::kInt32).to(tokens[0].device());
      }
      auto inputs_embeds = input_params[i].input_embedding;
      torch::Tensor h;
      if (inputs_embeds.defined()) {
        h = inputs_embeds;
      } else {
#if defined(USE_NPU)
        h = embed_tokens_[i](tokens[i], 0);
#else
        h = embed_tokens_[i](tokens[i]);
#endif
      }
      hs.push_back(std::move(h));
#if defined(USE_NPU)
      if (use_deepstack) {
        deep_stacks.push_back(
            input_params[i].deep_stacks);  // [num_deepstack, hidden_size]
      }
      auto target_cos_sin = atb_pos_embeds_[i](cos_sin_, positions[i], 0);
      auto target_cos_sin_chunks =
          target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
      auto cos_pos = target_cos_sin_chunks[0].contiguous();
      auto sin_pos = target_cos_sin_chunks[1].contiguous();

      if (positions[i].dim() == 2) {  // mrope
        auto apply = [this](torch::Tensor x) {
          auto freqs_t = x[0].clone();
          for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
            int64_t offset = dim_idx;
            int64_t section_len = mrope_section_[dim_idx];
            int64_t length = section_len * 3;
            auto idx_first_half =
                torch::arange(offset, length, 3, torch::kLong);
            auto idx_second_half =
                torch::arange(offset, length, 3, torch::kLong);
            auto idx_tensor =
                torch::cat({idx_first_half, idx_second_half}, 0).to(x.device());
            // freqs_t[..., idx] = freqs[dim_idx][..., idx]
            auto src = x[dim_idx].index_select(-1, idx_tensor);
            freqs_t.index_copy_(-1, idx_tensor, src);
          }
          return freqs_t;
        };
        cos_pos = apply(cos_pos.reshape(
            {positions[i].sizes().front(), -1, cos_pos.sizes().back()}));
        sin_pos = apply(sin_pos.reshape(
            {positions[i].sizes().front(), -1, sin_pos.sizes().back()}));
      }

      torch::Tensor attn_mask;

      torch::Tensor max_of_seq = torch::max(input_params[i].kv_seq_lens);
      max_seq_len_ = FLAGS_enable_chunked_prefill
                         ? std::max(max_of_seq.item<int>(), max_seq_len_)
                         : 128;
      attn_mask = attn_mask_.get_attn_mask(
          max_seq_len_, cos_pos.dtype().toScalarType(), cos_pos.device());

      if (FLAGS_enable_chunked_prefill) {
        int batch_size = input_params[i].q_seq_lens_vec.size();
        if (batch_size > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(batch_size);

          for (int j = 0; j < batch_size; j++) {
            int start = input_params[i].kv_seq_lens_vec[j] -
                        input_params[i].q_seq_lens_vec[j];
            int end = input_params[i].kv_seq_lens_vec[j];

            auto req_mask_slice = attn_mask.slice(0, start, end);
            req_mask_vec.emplace_back(req_mask_slice);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      }

      cos_poss.push_back(std::move(cos_pos));
      sin_poss.push_back(std::move(sin_pos));
      attn_masks.push_back(std::move(attn_mask));
#endif
    }
#if defined(USE_NPU)
    for (size_t i = 0; i < layers_.size(); i++) {
      std::vector<aclrtEvent*> events(micro_batch_num, nullptr);
      std::vector<std::atomic<bool>*> event_flags(micro_batch_num, nullptr);
      for (auto j = 0; j < micro_batch_num; ++j) {
        if (input_params[j].layer_synchronizer != nullptr) {
          events[j] = input_params[j].layer_synchronizer->get_event(i);
          event_flags[j] =
              input_params[j].layer_synchronizer->get_event_flag(i);
        }
      }
      auto& layer = layers_[i];

      layer(hs,
            cos_poss,
            sin_poss,
            attn_masks,
            kv_caches[i],
            input_params_news,
            i,
            events,
            event_flags);
      if (use_deepstack) {
        for (auto j = 0; j < micro_batch_num; ++j) {
          if (deep_stacks[j].size() > 0 && i < deep_stacks[j].size()) {
            hs[j] = deepstack_process(
                hs[j], input_params[j].visual_pos_masks, deep_stacks[j][i]);
          }
        }
      }
    }
    auto cancated_h = torch::cat(hs, 0);
    return norm_(cancated_h, 0);
#else
    bool is_prefill = input_params[0].q_max_seq_len > 1;
    auto attn_metadata =
        layer::AttentionMetadata::build(input_params[0], is_prefill);

    torch::Tensor h;
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(
          hs[0], positions[0], attn_metadata, kv_caches[i], input_params[0]);
    }
    return norm_(h);
#endif
  }

 private:
  torch::Tensor viusal_pos_mask_;
};
TORCH_MODULE(QWen3Model);

class QWen3ForCausalLMImpl : public LlmForCausalLMImplBase<QWen3Model> {
 public:
  QWen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen3Model>(context) {}
};
TORCH_MODULE(QWen3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL(qwen3, QWen3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS(qwen3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For qwen3/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", 4096);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
