

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

#include "qwen2_5_vl.h"

namespace xllm {

class Qwen2_5_VLForMMEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen2_5_VLForMMEmbeddingImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen2_5_VisionTransformer(context));
  }

  std::vector<int32_t> get_images_size(torch::Tensor image_grid_thw) {
    if (!image_grid_thw.defined()) return {};

    int32_t merge_size = model_args_.mm_image_merge_size();
    int32_t merge_length = merge_size * merge_size;

    std::vector<int32_t> images_size;
    int32_t count = image_grid_thw.size(0);
    images_size.reserve(count);
    for (int32_t idx = 0; idx < count; ++idx) {
      int32_t n_image_tokens =
          image_grid_thw[idx].prod().item<int32_t>() / merge_length;
      images_size.emplace_back(n_image_tokens);
    }
    return images_size;
  }

  MMDict encode(const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    const auto& mm_data = input_params.mm_data;

    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    std::optional<Qwen2_5_VLImageInputs> image_inputs;

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen2_5_VLImageInputs{pixel_values, image_grid_thw};

    CHECK(image_inputs.has_value());
    auto image_embeds = visual_(image_inputs->pixel_values.to(options_),
                                image_inputs->image_grid_thw,
                                input_params);

    std::vector<torch::Tensor> mm_embeddings;

    std::vector<int> image_sizes = get_images_size(image_grid_thw);
    mm_embeddings.reserve(image_sizes.size());

    int32_t token_start_idx = 0;
    for (int32_t image_size : image_sizes) {
      auto image_embed =
          image_embeds.slice(0, token_start_idx, token_start_idx + image_size);
      mm_embeddings.emplace_back(image_embed);
      token_start_idx += image_size;
    }
    CHECK(token_start_idx == image_embeds.size(0));
    MMDict mm_embeds;
    mm_embeds["image|embedding"] = mm_embeddings;
    return mm_embeds;
  };

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(state_dict->get_dict_with_prefix("visual."));
    }
    // verify
    visual_->verify_loaded_weights("visual.");
    visual_->merge_loaded_weights();
  }
  torch::Device device() const { return options_.device(); }
  const torch::TensorOptions& options() const { return options_; }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Qwen2_5_VisionTransformer visual_{nullptr};
};
TORCH_MODULE(Qwen2_5_VLForMMEmbedding);

template <>
class MMEmbeddingVLMImpl<xllm::Qwen2_5_VLForMMEmbedding>
    : public MMEmbeddingVLM {
 public:
  MMEmbeddingVLMImpl(xllm::Qwen2_5_VLForMMEmbedding model,
                     const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  MMDict encode(const ModelInputParams& input_params) override {
    return model_->encode(input_params);
  };

  torch::Tensor get_input_embeddings(const torch::Tensor& input_ids,
                                     const ModelInputParams& input_params) {
    return torch::Tensor{};
  }

  virtual torch::Tensor logits(const torch::Tensor& hidden_states,
                               const torch::Tensor& selected_idxes) {
    return torch::Tensor();
  }

  virtual torch::Tensor forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& input_params) {
    return torch::Tensor{};
  }
  virtual void prepare_expert_weight(int32_t layer_id,
                                     const std::vector<int32_t>& expert_ids) {
    return;
  }
  virtual void update_expert_weight(int32_t layer_id) { return; }
  virtual void set_npu_lm_head(layer::NpuLmHead& head) { return; }
  virtual layer::NpuLmHead get_npu_lm_head() { return nullptr; }
  virtual layer::NpuWordEmbedding get_npu_word_embedding() { return nullptr; }
  virtual void set_npu_word_embedding(layer::NpuWordEmbedding& embedding) {
    return;
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  torch::Device device() const override { return model_->device(); }

  const torch::TensorOptions& options() const override {
    return model_->options();
  }

 private:
  xllm::Qwen2_5_VLForMMEmbedding model_;
  torch::TensorOptions options_;
};

REGISTER_MM_EMBEDDING_VLM_MODEL_WITH_VARNAME(qwen2_5_vl_mm_embedding,
                                             qwen2_5_vl,
                                             Qwen2_5_VLForMMEmbedding);
}  // namespace xllm