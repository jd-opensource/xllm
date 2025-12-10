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

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/layers/lm_head.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "core/layers/qwen3_vision_encode_layer.h"
#include "models/llm/npu/qwen3_moe.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/qwen2_vl_image_processor.h"
#include "qwen2_5_vl.h"
#include "qwen3_vl.h"
#include "xllm_kernels/core/include/atb_speed/log.h"

namespace xllm {

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

class Qwen3_VLMoeForConditionalGenerationImpl : public torch::nn::Module {
 public:
  Qwen3_VLMoeForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen3_VisionTransformer(context));

    language_model_ =
        register_module("language_model", Qwen3MoeForCausalLM(context));
  }

  torch::Tensor get_input_embeddings(
      torch::Tensor input_ids,
      const std::optional<Qwen3_VLImageInputs>& image_input,
      const std::optional<Qwen3_VLVideoInputs>& video_input,
      const ModelInputParams& input_params) {
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (image_input) {
      // visual
      auto [image_embeds, deep_stacks] =
          visual_(image_input->pixel_values.to(options_),
                  image_input->image_grid_thw,
                  input_params);
      input_params.deep_stacks = deep_stacks;
      // merge
      auto is_multimodal = torch::isin(input_ids, model_args_.image_token_id());
      input_params.visual_pos_masks = is_multimodal;
      inputs_embeds.index_put_({is_multimodal}, image_embeds);
    }
    return inputs_embeds;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    const auto& mm_data = input_params.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();
    std::optional<Qwen3_VLImageInputs> image_inputs;
    std::optional<Qwen3_VLVideoInputs> video_inputs;

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen3_VLImageInputs{pixel_values, image_grid_thw};

    auto inputs_embeds =
        get_input_embeddings(tokens, image_inputs, video_inputs, input_params);
    input_params.input_embedding = inputs_embeds;
    auto emb = language_model_(tokens, positions, kv_caches, input_params);

    return emb;
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }

    // verify
    visual_->verify_loaded_weights("model.visual.");
    visual_->merge_loaded_weights();

    if (!model_args_.image_embedding_mode()) {
      language_model_->load_model(std::move(loader), "model.language_model.");
    }
  }

  layer::LmHead get_lm_head() { return language_model_->get_lm_head(); }
  void set_lm_head(layer::LmHead& head) { language_model_->set_lm_head(head); }

  layer::WordEmbedding get_word_embedding() {
    return language_model_->get_word_embedding();
  }

  void set_word_embedding(layer::WordEmbedding& word_embedding) {
    language_model_->set_word_embedding(word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  Qwen3_VisionTransformer visual_{nullptr};
  Qwen3MoeForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen3_VLMoeForConditionalGeneration);

REGISTER_INPUT_PROCESSOR(qwen3_vl_moe, Qwen2_5_VLInputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_vl_moe, Qwen3_VLMoeForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen3_vl_moe, Qwen2VLImageProcessor);
// register the model args
REGISTER_MODEL_ARGS(qwen3_vl_moe, [&] {
  // text config
  LOAD_ARG_OR(model_type, "model_type", "qwen3_vl_moe");
  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 151643);
  LOAD_ARG_OR(decoder_sparse_step, "text_config.decoder_sparse_step", 1);
  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 151645);
  LOAD_ARG_OR_FUNC(head_dim, "text_config.head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 2048);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 5632);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 128000);
  // LOAD_ARG(mlp_only_layers, "text_config.mlp_only_layers");
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 1408);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 16);
  LOAD_ARG_OR(num_experts, "text_config.num_experts", 128);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 24);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 16);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(rope_scaling_rope_type, "text_config.rope_scaling.type", "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "text_config.rope_scaling.mrope_section");
  // LOAD_ARG_OR(rope_scaling_mrope_interleaved,"text_config.rope_scaling.mrope_interleaved",true);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 5000000.0f);
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 151936);

  // vision config
  LOAD_ARG(mm_deepstack_visual_indexes,
           "vision_config.deepstack_visual_indexes");
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 27);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_position_embeddings,
              "vision_config.num_position_embeddings",
              2304);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 3584);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 16);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });

  LOAD_ARG_OR(image_token_id, "image_token_id", 151655);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(video_token_id, "video_token_id", 151656);
  LOAD_ARG_OR(vision_end_token_id, "vision_end_token_id", 151653);
  LOAD_ARG_OR(vision_start_token_id, "vision_start_token_id", 151652);
});
}  // namespace xllm
