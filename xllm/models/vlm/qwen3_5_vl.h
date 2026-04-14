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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define XLLM_DISABLE_GENERIC_VLM_REGISTRATION
#include "models/vlm/qwen3_vl.h"
#undef XLLM_DISABLE_GENERIC_VLM_REGISTRATION

#if defined(USE_NPU)
#include "models/vlm/npu/qwen3_vl.h"
#endif

#include "models/llm/qwen3_5.h"

namespace xllm {

using Qwen3_5_VLInputProcessor = Qwen3_VLInputProcessor;
using Qwen3_5_VLImageProcessor = Qwen3VLImageProcessor;

class Qwen3_5_VLForConditionalGenerationImpl : public torch::nn::Module {
 public:
  explicit Qwen3_5_VLForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
#if defined(USE_NPU)
    visual_ =
        register_module("visual", npu::model::Qwen3_VisionTransformer(context));
#else
    visual_ = register_module("visual", Qwen3_VisionTransformer(context));
#endif
    language_model_ =
        register_module("language_model", Qwen3_5ForCausalLM(context));
  }

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Qwen3_VLImageInputs>& image_inputs,
                             std::optional<Qwen3_VLVideoInputs>& video_inputs) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values")) {
      pixel_values = res.value();
    }

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos")) {
      pixel_values_videos = res.value();
    }

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw")) {
      video_grid_thw = res.value();
    }

    if (pixel_values.defined() && image_grid_thw.defined()) {
      image_inputs = Qwen3_VLImageInputs{pixel_values, image_grid_thw};
    }

    if (pixel_values_videos.defined() && video_grid_thw.defined()) {
      video_inputs = Qwen3_VLVideoInputs{pixel_values_videos, video_grid_thw};
    }
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen3_VLImageInputs> image_input;
    std::optional<Qwen3_VLVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);

    MMDict multimodal_embeds;
    const int32_t merge_size = model_args_.mm_image_merge_size();
    if (image_input) {
      torch::Tensor image_embeds;
      std::vector<torch::Tensor> deep_stacks;
      std::tie(image_embeds, deep_stacks) =
          visual_(image_input->pixel_values.to(options_),
                  image_input->image_grid_thw.to(options_.device()),
                  input_params);

      auto image_tokens =
          (image_input->image_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> image_tokens_vec(
          image_tokens.data_ptr<int64_t>(),
          image_tokens.data_ptr<int64_t>() + image_tokens.numel());
      multimodal_embeds["image|embedding"] =
          image_embeds.split(image_tokens_vec, 0);

      for (size_t i = 0; i < deep_stacks.size(); ++i) {
        multimodal_embeds[std::string("image|embedding|deepstack_") +
                          std::to_string(i)] =
            deep_stacks[i].split(image_tokens_vec, 0);
      }
    }

    if (video_input) {
      torch::Tensor video_embeds;
      std::vector<torch::Tensor> deep_stacks;
      std::tie(video_embeds, deep_stacks) =
          visual_(video_input->pixel_values_videos.to(options_),
                  video_input->video_grid_thw.to(options_.device()),
                  input_params);

      auto video_tokens =
          (video_input->video_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);

      std::vector<int64_t> video_tokens_vec(
          video_tokens.data_ptr<int64_t>(),
          video_tokens.data_ptr<int64_t>() + video_tokens.numel());
      multimodal_embeds["video|embedding"] =
          video_embeds.split(video_tokens_vec, 0);

      for (size_t i = 0; i < deep_stacks.size(); ++i) {
        multimodal_embeds[std::string("video|embedding|deepstack_") +
                          std::to_string(i)] =
            deep_stacks[i].split(video_tokens_vec, 0);
      }
    }
    return multimodal_embeds;
  }

  torch::Tensor generate_multimodal_mask(torch::Tensor input_ids) {
    torch::Tensor special_token_ids = torch::tensor(
        {model_args_.image_token_id(), model_args_.video_token_id()},
        input_ids.options().dtype(torch::kInt64));
    return torch::isin(input_ids, special_token_ids);
  }

  std::vector<torch::Tensor> get_deep_stacks(
      const ModelInputParams& input_params) {
    const auto& mm_data = input_params.mm_data;
    if (!mm_data.has("embedding|deepstack_0")) {
      return {};
    }

    std::vector<torch::Tensor> deepstacks = {
        mm_data.get<torch::Tensor>("embedding|deepstack_0").value(),
        mm_data.get<torch::Tensor>("embedding|deepstack_1").value(),
        mm_data.get<torch::Tensor>("embedding|deepstack_2").value()};
    return deepstacks;
  }

  torch::Tensor merge_multimodal_embeddings(
      torch::Tensor inputs_embeds,
      const torch::Tensor& multimodal_embeds,
      const torch::Tensor& is_multimodal) {
    inputs_embeds.index_put_({is_multimodal}, multimodal_embeds);
    return inputs_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor multimodal_embeds;
    if (const auto& emb = mm_data.get<torch::Tensor>("embedding")) {
      multimodal_embeds = emb.value();
    }

    torch::Tensor inputs_embeds =
        language_model_->get_word_embedding()(input_ids);
    if (!multimodal_embeds.defined()) {
      return inputs_embeds;
    }

    torch::Tensor is_multimodal = generate_multimodal_mask(input_ids);
    input_params.visual_pos_masks = is_multimodal;
    return merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeds, is_multimodal);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    input_params.deep_stacks = std::move(get_deep_stacks(input_params));
    return language_model_(tokens, positions, kv_caches, input_params);
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
#if defined(USE_NPU)
    visual_->verify_loaded_weights("model.visual.");
    visual_->merge_loaded_weights();
#endif

    if (!model_args_.image_embedding_mode()) {
      language_model_->load_model(std::move(loader));
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
#if defined(USE_NPU)
  npu::model::Qwen3_VisionTransformer visual_{nullptr};
#else
  Qwen3_VisionTransformer visual_{nullptr};
#endif
  Qwen3_5ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen3_5_VLForConditionalGeneration);

#define LOAD_QWEN3_5_VL_TEXT_OR_ROOT(arg_name, json_key, default_value) \
  LOAD_ARG_OR(arg_name, "text_config." json_key, default_value);        \
  LOAD_ARG_OR(arg_name, json_key, args->arg_name())

#define LOAD_QWEN3_5_VL_ROPE_ARG(arg_name, default_value)                    \
  LOAD_ARG_OR(arg_name, "text_config." #arg_name, default_value);            \
  LOAD_ARG_OR(arg_name, #arg_name, args->arg_name());                        \
  LOAD_ARG_OR(                                                               \
      arg_name, "text_config.rope_scaling." #arg_name, args->arg_name());    \
  LOAD_ARG_OR(arg_name, "rope_scaling." #arg_name, args->arg_name());        \
  LOAD_ARG_OR(                                                               \
      arg_name, "text_config.rope_parameters." #arg_name, args->arg_name()); \
  LOAD_ARG_OR(arg_name, "rope_parameters." #arg_name, args->arg_name())

#define LOAD_QWEN3_5_VL_NESTED_ROPE_ARG(arg_name, json_key, default_value)  \
  LOAD_ARG_OR(arg_name, "text_config." #arg_name, default_value);           \
  LOAD_ARG_OR(arg_name, #arg_name, args->arg_name());                       \
  LOAD_ARG_OR(                                                              \
      arg_name, "text_config.rope_scaling." json_key, args->arg_name());    \
  LOAD_ARG_OR(arg_name, "rope_scaling." json_key, args->arg_name());        \
  LOAD_ARG_OR(                                                              \
      arg_name, "text_config.rope_parameters." json_key, args->arg_name()); \
  LOAD_ARG_OR(arg_name, "rope_parameters." json_key, args->arg_name())

#define LOAD_QWEN3_5_VL_TOKEN_ARG(arg_name, default_value)        \
  LOAD_ARG_OR(arg_name, "text_config." #arg_name, default_value); \
  LOAD_ARG_OR(arg_name, #arg_name, args->arg_name())

REGISTER_INPUT_PROCESSOR(qwen3_5_vl, Qwen3_5_VLInputProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_5_vl, Qwen3_5_VLForConditionalGeneration);
REGISTER_IMAGE_PROCESSOR(qwen3_5_vl, Qwen3_5_VLImageProcessor);

REGISTER_MODEL_ARGS(qwen3_5_vl, [&] {
  SET_ARG(model_type, "qwen3_5_vl");

  LOAD_ARG_OR(dtype, "text_config.dtype", "bfloat16");
  LOAD_ARG_OR(dtype, "dtype", args->dtype());
  LOAD_ARG_OR(dtype, "text_config.torch_dtype", args->dtype());
  LOAD_ARG_OR(dtype, "torch_dtype", args->dtype());

  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(attention_bias, "attention_bias", false);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(attention_dropout, "attention_dropout", 0.0f);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(bos_token_id, "bos_token_id", 151643);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(eos_token_id, "eos_token_id", 151645);
  LOAD_QWEN3_5_VL_TOKEN_ARG(vision_start_token_id, 248053);
  LOAD_QWEN3_5_VL_TOKEN_ARG(vision_end_token_id, 248054);
  LOAD_QWEN3_5_VL_TOKEN_ARG(vision_token_id, 248055);
  LOAD_QWEN3_5_VL_TOKEN_ARG(image_token_id, 248056);
  LOAD_QWEN3_5_VL_TOKEN_ARG(video_token_id, 248057);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(head_dim, "head_dim", 256);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(hidden_act, "hidden_act", "silu");
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(hidden_size, "hidden_size", 5120);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(initializer_range, "initializer_range", 0.02f);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(intermediate_size, "intermediate_size", 17408);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(
      max_position_embeddings, "max_position_embeddings", 262144);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(max_window_layers, "max_window_layers", 64);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(n_heads, "num_attention_heads", 24);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(n_layers, "num_hidden_layers", 64);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 4);
  LOAD_ARG_OR(
      n_kv_heads, "num_key_value_heads", args->n_kv_heads().value_or(4));
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_QWEN3_5_VL_ROPE_ARG(rope_theta, 10000000.0f);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(use_sliding_window, "use_sliding_window", false);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(sliding_window, "sliding_window", 4096);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(
      tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(vocab_size, "vocab_size", 248320);
  LOAD_ARG_OR(
      mlp_only_layers, "text_config.mlp_only_layers", std::vector<int32_t>());
  LOAD_ARG_OR(mlp_only_layers, "mlp_only_layers", args->mlp_only_layers());
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(attn_output_gate, "attn_output_gate", true);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(
      full_attention_interval, "full_attention_interval", 4);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(
      linear_conv_kernel_dim, "linear_conv_kernel_dim", 4);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(linear_key_head_dim, "linear_key_head_dim", 128);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(
      linear_num_key_heads, "linear_num_key_heads", 16);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(
      linear_num_value_heads, "linear_num_value_heads", 48);
  LOAD_QWEN3_5_VL_TEXT_OR_ROOT(
      linear_value_head_dim, "linear_value_head_dim", 128);
  LOAD_QWEN3_5_VL_ROPE_ARG(partial_rotary_factor, 0.25f);
  LOAD_ARG_OR(rope_scaling_mrope_interleaved,
              "text_config.rope_scaling.mrope_interleaved",
              false);
  LOAD_ARG_OR(rope_scaling_mrope_interleaved,
              "rope_scaling.mrope_interleaved",
              args->rope_scaling_mrope_interleaved());
  LOAD_ARG_OR(rope_scaling_mrope_interleaved,
              "text_config.rope_parameters.mrope_interleaved",
              args->rope_scaling_mrope_interleaved());
  LOAD_ARG_OR(rope_scaling_mrope_interleaved,
              "rope_parameters.mrope_interleaved",
              args->rope_scaling_mrope_interleaved());
  LOAD_ARG_OR(num_nextn_predict_layers, "text_config.mtp_num_hidden_layers", 0);
  LOAD_ARG_OR(num_nextn_predict_layers,
              "mtp_num_hidden_layers",
              args->num_nextn_predict_layers());
  LOAD_ARG_OR(num_nextn_predict_layers,
              "text_config.num_nextn_predict_layers",
              args->num_nextn_predict_layers());
  LOAD_ARG_OR(num_nextn_predict_layers,
              "num_nextn_predict_layers",
              args->num_nextn_predict_layers());
  LOAD_ARG_OR(
      layer_types, "text_config.layer_types", std::vector<std::string>());
  LOAD_ARG_OR(layer_types, "layer_types", args->layer_types());
  LOAD_ARG_OR(
      layer_types, "text_config.layers_block_type", args->layer_types());
  LOAD_ARG_OR(layer_types, "layers_block_type", args->layer_types());

  SET_ARG(moe_intermediate_size, 0);
  SET_ARG(norm_topk_prob, true);
  SET_ARG(num_experts, 0);
  SET_ARG(num_experts_per_tok, 0);
  SET_ARG(output_router_logits, false);
  SET_ARG(router_aux_loss_coef, 0.001f);
  SET_ARG(shared_expert_intermediate_size, 0);
  SET_ARG(n_routed_experts, 0);
  SET_ARG(n_shared_experts, 0);
  SET_ARG(scoring_func, "softmax");
  SET_ARG(topk_method, "");
  SET_ARG(n_group, -1);
  SET_ARG(topk_group, 0);
  SET_ARG(routed_scaling_factor, 1.0f);
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));

  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.depth", 27);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "gelu_pytorch_tanh");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1152);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4304);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_heads", 16);
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_channels", 3);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.out_hidden_size", 5120);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 16);
  LOAD_ARG_OR(mm_num_position_embeddings,
              "vision_config.num_position_embeddings",
              2304);
  LOAD_ARG_OR(mm_spatial_merge_size, "vision_config.spatial_merge_size", 2);
  LOAD_ARG(mm_deepstack_visual_indexes,
           "vision_config.deepstack_visual_indexes");
  LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size", 2);
  LOAD_ARG_OR_FUNC(mm_head_dim, "head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });
  LOAD_QWEN3_5_VL_NESTED_ROPE_ARG(rope_scaling_rope_type, "type", "default");
  LOAD_QWEN3_5_VL_NESTED_ROPE_ARG(rope_scaling_mrope_section,
                                  "mrope_section",
                                  std::vector<int64_t>({11, 11, 10}));
  if (args->rope_scaling_rope_type() == "default") {
    args->rope_scaling_rope_type() = "mrope";
  }
});

#undef LOAD_QWEN3_5_VL_TOKEN_ARG
#undef LOAD_QWEN3_5_VL_NESTED_ROPE_ARG
#undef LOAD_QWEN3_5_VL_ROPE_ARG
#undef LOAD_QWEN3_5_VL_TEXT_OR_ROOT

}  // namespace xllm
