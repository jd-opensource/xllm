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

#include "core/framework/model/model_output.h"
#include "core/framework/request/mm_data_item.h"
#include "core/layers/common/lm_head.h"
#include "models/model_registry.h"

namespace xllm {

struct Qwen3_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct Qwen3_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

template <typename VisionTransformer, typename LanguageModel>
class Qwen3VLForConditionalGenerationBase : public torch::nn::Module {
 public:
  Qwen3VLForConditionalGenerationBase(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", VisionTransformer(context));
    language_model_ = register_module("language_model", LanguageModel(context));
  }

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Qwen3_VLImageInputs>& image_inputs,
                             std::optional<Qwen3_VLVideoInputs>& video_inputs) {
    const auto& mm_data = input_params.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos"))
      pixel_values_videos = res.value();

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen3_VLImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined())
      video_inputs = Qwen3_VLVideoInputs{pixel_values_videos, video_grid_thw};
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen3_VLImageInputs> image_input;
    std::optional<Qwen3_VLVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);

    MMDict multimodal_embeds;
    auto merge_size = model_args_.mm_image_merge_size();
    if (image_input) {
      auto [image_embeds, deep_stacks] =
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
          image_embeds.split(image_tokens_vec, 0 /*dim*/);

      for (size_t i = 0; i < deep_stacks.size(); ++i) {
        multimodal_embeds[std::string("image|embedding|deepstack_") +
                          std::to_string(i)] =
            deep_stacks[i].split(image_tokens_vec, 0 /*dim*/);
      }
    }
    if (video_input) {
      auto [video_embeds, deep_stacks] =
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
          video_embeds.split(video_tokens_vec, 0 /*dim*/);

      for (size_t i = 0; i < deep_stacks.size(); ++i) {
        multimodal_embeds[std::string("video|embedding|deepstack_") +
                          std::to_string(i)] =
            deep_stacks[i].split(video_tokens_vec, 0 /*dim*/);
      }
    }
    return multimodal_embeds;
  }

  torch::Tensor generate_multimodal_mask(torch::Tensor input_ids) {
    auto special_token_ids = torch::tensor(
        {model_args_.image_token_id(), model_args_.video_token_id()},
        input_ids.options().dtype(torch::kInt64));
    auto is_multimodal = torch::isin(input_ids, special_token_ids);
    return is_multimodal;
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
    auto inputs_embeds = language_model_->get_input_embeddings(input_ids);
    if (!multimodal_embeds.defined()) {
      return inputs_embeds;
    }
    auto is_multimodal = generate_multimodal_mask(input_ids);
    input_params.visual_pos_masks = is_multimodal;
    inputs_embeds = merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeds, is_multimodal);
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    input_params.deep_stacks = std::move(get_deep_stacks(input_params));
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->pooler(hidden_states, seleted_idxes);
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

 protected:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  VisionTransformer visual_{nullptr};
  LanguageModel language_model_{nullptr};
};

}  // namespace xllm
