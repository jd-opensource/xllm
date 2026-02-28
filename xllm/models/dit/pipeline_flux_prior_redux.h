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
#include "pipeline_flux_base.h"
#include "processors/siglip_image_processor.h"
#include "siglip_vision_model.h"
// pipeline_flux_prior_redux compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_prior_redux.py

namespace xllm {

class ReduxImageEncoderImpl : public torch::nn::Module {
 public:
  explicit ReduxImageEncoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    act_ = register_module("act", torch::nn::Functional(torch::silu));

    redux_up_ =
        register_module("redux_up",
                        layer::AddMatmul(model_args.mm_hidden_size(),
                                         model_args.mm_intermediate_size() * 3,
                                         true,
                                         options));
    redux_down_ =
        register_module("redux_down",
                        layer::AddMatmul(model_args.mm_intermediate_size() * 3,
                                         model_args.mm_intermediate_size(),
                                         true,
                                         options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return redux_down_(act_(redux_up_(hidden_states)));
  }

  void load_model(std::unique_ptr<DiTFolderLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      redux_up_->load_state_dict(state_dict->get_dict_with_prefix("redux_up."));
      redux_down_->load_state_dict(
          state_dict->get_dict_with_prefix("redux_down."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    redux_up_->verify_loaded_weights(prefix + "redux_up.");
    redux_up_->verify_loaded_weights(prefix + "redux_down.");
  }

 private:
  layer::AddMatmul redux_up_{nullptr};
  layer::AddMatmul redux_down_{nullptr};

  torch::nn::Functional act_ = nullptr;
};
TORCH_MODULE(ReduxImageEncoder);

REGISTER_MODEL_ARGS(ReduxImageEncoder, [&] {
  LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
  LOAD_ARG_OR(mm_hidden_size, "redux_dim", 1152);
  LOAD_ARG_OR(mm_intermediate_size, "txt_in_features", 4096);
});

class FluxPriorReduxPipelineImpl : public FluxPipelineBaseImpl {
 public:
  explicit FluxPriorReduxPipelineImpl(const DiTModelContext& context) {
    auto model_args = context.get_model_args("feature_extractor");
    options_ = context.get_tensor_options();
    image_encoder_ =
        SiglipVisionModel(context.get_model_context("image_encoder"));
    image_embedder_ =
        ReduxImageEncoder(context.get_model_context("image_embedder"));
    feature_extractor_ = std::make_unique<SiglipImageProcessor>(model_args);
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    std::string model_path = loader->model_root_path();
    auto image_encoder_loader = loader->take_component_loader("image_encoder");
    auto image_embedder_loader =
        loader->take_component_loader("image_embedder");
    image_encoder_->load_model(std::move(image_encoder_loader));
    image_encoder_->to(options_.device());
    image_embedder_->load_model(std::move(image_embedder_loader));
    image_embedder_->to(options_.device());
  }

  torch::Tensor encode_image(const torch::Tensor& image,
                             int64_t num_images_per_prompt) {
    auto imgs = feature_extractor_->preprocess(image).to(options_);
    auto image_enc_hidden_states = image_encoder_->forward(imgs);
    image_enc_hidden_states =
        image_enc_hidden_states.repeat_interleave(num_images_per_prompt, 0);
    return image_enc_hidden_states;
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;
    auto image = input.images.defined() ? std::make_optional(input.images)
                                        : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto pooled_prompt_embeds =
        input.pooled_prompt_embeds.defined()
            ? std::make_optional(input.pooled_prompt_embeds)
            : std::nullopt;
    auto prompt_embeds_scale = generation_params.prompt_embeds_scale;
    auto pooled_prompt_embeds_scale =
        generation_params.pooled_prompt_embeds_scale;
    std::vector<torch::Tensor> output =
        forward_impl(image.value(),
                     prompt_embeds,
                     pooled_prompt_embeds,
                     generation_params.height,
                     generation_params.width,
                     prompt_embeds_scale,
                     pooled_prompt_embeds_scale);
    DiTForwardOutput out;
    out.tensors = output;
    return out;
  }

  std::vector<torch::Tensor> forward_impl(
      torch::Tensor image,
      std::optional<torch::Tensor> prompt_embeds_opt,
      std::optional<torch::Tensor> pooled_prompt_embeds_opt,
      int64_t height = 384,
      int64_t width = 384,
      float prompt_embeds_scale = 1.0f,
      float pooled_prompt_embeds_scale = 1.0f) {
    torch::NoGradGuard no_grad;
    int64_t batch_size = image.dim() == 4 ? image.size(0) : 1;
    torch::Tensor image_latents =
        encode_image(image, /*num_images_per_prompt=*/1);
    torch::Tensor image_embeds =
        image_embedder_->forward(image_latents).to(options_);

    // prompt_embeds: [batch_size, seq_len, hidden_dim]
    torch::Tensor prompt_embeds = prompt_embeds_opt.value_or(torch::zeros(
        {batch_size, /*seq_len=*/512, /*hidden_dim=*/4096}, options_));
    // pooled_prompt_embeds: [batch_size, pooled_hidden_dim]
    torch::Tensor pooled_prompt_embeds = pooled_prompt_embeds_opt.value_or(
        torch::zeros({batch_size, /*pooled_hidden_dim=*/768}, options_));

    prompt_embeds = torch::cat({prompt_embeds, image_embeds}, /*dim=*/1);
    prompt_embeds *= torch::full({batch_size}, prompt_embeds_scale, options_)
                         .view({-1, 1, 1});
    pooled_prompt_embeds *=
        torch::full({batch_size}, pooled_prompt_embeds_scale, options_)
            .view({-1, 1});

    prompt_embeds = torch::sum(prompt_embeds, /*dim=*/0, /*keepdim=*/true);
    pooled_prompt_embeds =
        torch::sum(pooled_prompt_embeds, /*dim=*/0, /*keepdim=*/true);

    // TODO: Add tensor output interface.
    return {prompt_embeds, pooled_prompt_embeds};
  }

 private:
  SiglipVisionModel image_encoder_{nullptr};
  std::unique_ptr<SiglipImageProcessor> feature_extractor_;
  ReduxImageEncoder image_embedder_{nullptr};
};
TORCH_MODULE(FluxPriorReduxPipeline);

REGISTER_DIT_MODEL(fluxredux, FluxPriorReduxPipeline);
}  // namespace xllm
