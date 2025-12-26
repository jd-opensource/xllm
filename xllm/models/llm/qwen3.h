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
    auto parallel_args = context.get_parallel_args();
    auto dp_local_tp_size =
        parallel_args.world_size() / parallel_args.dp_size();
    dp_rank_ = parallel_args.rank() / dp_local_tp_size;
    if (!mrope_section_.empty()) {
      cos_sin_ = layer::rotary::get_concat_rotary_embedding(
          128,
          model_args.max_position_embeddings(),
          model_args.rope_theta(),
          options);
    }

    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::RMSNorm(context));
    embed_tokens_ =
        register_module("embed_tokens", layer::WordEmbedding(context));
    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto layer = QWen3DecoderLayer(context);
      layers_.push_back(layer);
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

  std::pair<torch::Tensor, torch::Tensor> apply_mrope(
      const torch::Tensor positions) override {
    auto target_cos_sin = cos_sin_.index({positions});
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    auto apply = [this](torch::Tensor x) {
      auto freqs_t = x[0].clone();
      for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
        int64_t offset = dim_idx;
        int64_t section_len = mrope_section_[dim_idx];
        int64_t length = section_len * 3;
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

  virtual torch::Tensor forward(torch::Tensor tokens,
                                torch::Tensor positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& input_params) {
    bool use_deepstack = input_params.deep_stacks.size() > 0;
    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    std::vector<torch::Tensor> deep_stacks;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = embed_tokens_(tokens);
    }
    if (use_deepstack) {
      deep_stacks = input_params.deep_stacks;  // [num_deepstack, hidden_size]
    }

    auto& dp_token_nums = input_params_new.dp_global_token_nums;
    std::replace(dp_token_nums.begin(), dp_token_nums.end(), 0, 1);
    auto attn_metadata = layer::AttentionMetadata::build(input_params_new);
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
                input_params_new);

      if (use_deepstack) {
        if (deep_stacks.size() > 0 && i < deep_stacks.size()) {
          h = deepstack_process(
              h, input_params.visual_pos_masks, deep_stacks[i]);
        }
      }
    }
    return std::get<0>(norm_(h, residual));
  }
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
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm
