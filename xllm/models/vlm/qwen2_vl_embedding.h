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

#include "core/framework/model/embedding_vlm.h"
#include "models/llm/qwen2.h"
#include "models/vlm/qwen2_vl.h"

namespace xllm {

class Qwen2_VLForEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen2_VLForEmbeddingImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_VisionTransformer(context));
    language_model_ =
        register_module("language_model", QWen2ForCausalLM(context));
  }

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Qwen2_VLImageInputs>& image_inputs,
                             std::optional<Qwen2_VLVideoInputs>& video_inputs) {
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

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen2_VLImageInputs{pixel_values, image_grid_thw};
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen2_VLImageInputs> image_input;
    std::optional<Qwen2_VLVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);
    auto merge_size = model_args_.mm_image_merge_size();
    MMDict multimodal_embeds;
    if (image_input) {
      // visual
      auto image_embeds = visual_(image_input->pixel_values.to(options_),
                                  image_input->image_grid_thw,
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
    inputs_embeds = merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeds, is_multimodal);
    return inputs_embeds;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    auto emb = language_model_(tokens, positions, kv_caches, input_params);
    return emb;
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    auto pooler_output = torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    return pooler_output;
  }

  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) {
    LOG(ERROR) << "logits() not implemented for Embedding Model!";
    return torch::empty({0});
  }

  torch::Device device() const { return options_.device(); }

  const torch::TensorOptions& options() const { return options_; }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix("visual."));
    }
    // if (!model_args_.image_embedding_mode()) {
    language_model_->load_model(std::move(loader));
    // }
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

  Qwen2_VisionTransformer visual_{nullptr};
  QWen2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen2_VLForEmbedding);

template <>
class EmbeddingVLMImpl<xllm::Qwen2_VLForEmbedding> : public EmbeddingVLM {
 public:
  EmbeddingVLMImpl(xllm::Qwen2_VLForEmbedding model,
                   const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  MMDict encode(const ModelInputParams& input_params) override {
    return model_->encode(input_params);
  };
  torch::Tensor get_input_embeddings(const torch::Tensor& input_ids,
                                     const ModelInputParams& input_params) {
    return torch::Tensor{};
  }
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& parameters) override {
    return model_->forward(tokens, positions, kv_caches, parameters);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->logits(hidden_states, seleted_idxes);
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    return model_->pooler(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  torch::Device device() const override { return model_->device(); }

  const torch::TensorOptions& options() const override {
    return model_->options();
  }

  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }

  // Delegate head/embedding accessors to underlying model implementation.
  layer::LmHead get_lm_head() override { return model_->get_lm_head(); }
  void set_lm_head(layer::LmHead& head) override { model_->set_lm_head(head); }
  layer::WordEmbedding get_word_embedding() override {
    return model_->get_word_embedding();
  }
  void set_word_embedding(layer::WordEmbedding& embedding) override {
    model_->set_word_embedding(embedding);
  }

 private:
  xllm::Qwen2_VLForEmbedding model_;
  torch::TensorOptions options_;
};

REGISTER_EMBEDDING_VLM_MODEL_WITH_VARNAME(qwen2_vl_embedding,
                                          qwen2_vl,
                                          Qwen2_VLForEmbedding);
}  // namespace xllm