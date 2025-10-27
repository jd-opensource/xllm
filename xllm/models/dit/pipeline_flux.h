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
#include "core/layers/pos_embedding.h"
#include "core/layers/rotary_embedding.h"
#include "dit.h"
#include "pipeline_flux_base.h"
// pipeline_flux compatible with huggingface weights
// ref to:
// https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

namespace xllm {

class FluxPipelineImpl : public FluxPipelineBaseImpl {
 public:
  FluxPipelineImpl(const DiTModelContext& context) {
    const auto& model_args = context.get_model_args("vae");
    options_ = context.get_tensor_options();
    vae_scale_factor_ = 1 << (model_args.block_out_channels().size() - 1);
    device_ = options_.device();
    dtype_ = options_.dtype().toScalarType();

    vae_shift_factor_ = model_args.shift_factor();
    vae_scaling_factor_ = model_args.scale_factor();
    default_sample_size_ = 128;
    tokenizer_max_length_ = 77;  // TODO: get from config file
    LOG(INFO) << "Initializing Flux pipeline...";
    vae_image_processor_ = VAEImageProcessor(
        context.get_model_context("vae"), true, true, false, false, false);
    vae_ = VAE(context.get_model_context("vae"));
    LOG(INFO) << "VAE initialized.";
    pos_embed_ = register_module(
        "pos_embed",
        FluxPosEmbed(10000,
                     context.get_model_args("transformer").axes_dims_rope()));
    transformer_ = FluxDiTModel(context.get_model_context("transformer"));
    LOG(INFO) << "DiT transformer initialized.";
    t5_ = T5EncoderModel(context.get_model_context("text_encoder_2"));
    LOG(INFO) << "T5 initialized.";
    clip_text_model_ = CLIPTextModel(context.get_model_context("text_encoder"));
    LOG(INFO) << "CLIP text model initialized.";
    scheduler_ =
        FlowMatchEulerDiscreteScheduler(context.get_model_context("scheduler"));
    LOG(INFO) << "Flux pipeline initialized.";
    register_module("vae", vae_);
    LOG(INFO) << "VAE registered.";
    register_module("vae_image_processor", vae_image_processor_);
    LOG(INFO) << "VAE image processor registered.";
    register_module("transformer", transformer_);
    LOG(INFO) << "DiT transformer registered.";
    register_module("t5", t5_);
    LOG(INFO) << "T5 registered.";
    register_module("scheduler", scheduler_);
    LOG(INFO) << "Scheduler registered.";
    register_module("clip_text_model", clip_text_model_);
    LOG(INFO) << "CLIP text model registered.";
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    const auto& generation_params = input.generation_params;

    auto seed = generation_params.seed > 0 ? generation_params.seed : 42;
    auto prompts = std::make_optional(input.prompts);
    auto prompts_2 = input.prompts_2.empty()
                         ? std::nullopt
                         : std::make_optional(input.prompts_2);
    auto negative_prompts = input.negative_prompts.empty()
                                ? std::nullopt
                                : std::make_optional(input.negative_prompts);
    auto negative_prompts_2 =
        input.negative_prompts_2.empty()
            ? std::nullopt
            : std::make_optional(input.negative_prompts_2);

    auto latents = input.latents.defined() ? std::make_optional(input.latents)
                                           : std::nullopt;
    auto prompt_embeds = input.prompt_embeds.defined()
                             ? std::make_optional(input.prompt_embeds)
                             : std::nullopt;
    auto negative_prompt_embeds =
        input.negative_prompt_embeds.defined()
            ? std::make_optional(input.negative_prompt_embeds)
            : std::nullopt;
    auto pooled_prompt_embeds =
        input.pooled_prompt_embeds.defined()
            ? std::make_optional(input.pooled_prompt_embeds)
            : std::nullopt;
    auto negative_pooled_prompt_embeds =
        input.negative_pooled_prompt_embeds.defined()
            ? std::make_optional(input.negative_pooled_prompt_embeds)
            : std::nullopt;

    std::vector<torch::Tensor> output = forward_(
        prompts,                                       // prompt
        prompts_2,                                     // prompt_2
        negative_prompts,                              // negative_prompt
        negative_prompts_2,                            // negative_prompt_2
        generation_params.true_cfg_scale,              // cfg scale
        std::make_optional(generation_params.height),  // height
        std::make_optional(generation_params.width),   // width
        generation_params.num_inference_steps,         // num_inference_steps
        generation_params.guidance_scale,              // guidance_scale
        generation_params.num_images_per_prompt,       // num_images_per_prompt
        seed,                                          // seed
        latents,                                       // latents
        prompt_embeds,                                 // prompt_embeds
        negative_prompt_embeds,                        // negative_prompt_embeds
        pooled_prompt_embeds,                          // pooled_prompt_embeds
        negative_pooled_prompt_embeds,         // negative_pooled_prompt_embeds
        generation_params.max_sequence_length  // max_sequence_length
    );

    DiTForwardOutput out;
    out.tensors = torch::chunk(output[0], output[0].size(0), 0);
    LOG(INFO) << "Output tensor chunks size: " << out.tensors.size();
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "FluxPipeline loading model from" << loader->model_root_path();
    std::string model_path = loader->model_root_path();
    auto transformer_loader = loader->take_component_loader("transformer");
    auto vae_loader = loader->take_component_loader("vae");
    auto t5_loader = loader->take_component_loader("text_encoder_2");
    auto clip_loader = loader->take_component_loader("text_encoder");
    auto tokenizer_loader = loader->take_component_loader("tokenizer");
    auto tokenizer_2_loader = loader->take_component_loader("tokenizer_2");
    LOG(INFO)
        << "Flux model components loaded, start to load weights to sub models";
    transformer_->load_model(std::move(transformer_loader));
    transformer_->to(device_);
    vae_->load_model(std::move(vae_loader));
    vae_->to(device_);
    t5_->load_model(std::move(t5_loader));
    t5_->to(device_);
    clip_text_model_->load_model(std::move(clip_loader));
    clip_text_model_->to(device_);
    tokenizer_ = tokenizer_loader->tokenizer();
    tokenizer_2_ = tokenizer_2_loader->tokenizer();
  }

 private:
  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      int64_t seed,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));
    std::vector<int64_t> shape = {
        batch_size, num_channels_latents, adjusted_height, adjusted_width};
    if (latents.has_value()) {
      torch::Tensor latent_image_ids = prepare_latent_image_ids(
          batch_size, adjusted_height / 2, adjusted_width / 2);
      return {latents.value(), latent_image_ids};
    }
    torch::Tensor latents_tensor = randn_tensor(shape, seed, options_);
    torch::Tensor packed_latents = pack_latents(latents_tensor,
                                                batch_size,
                                                num_channels_latents,
                                                adjusted_height,
                                                adjusted_width);
    torch::Tensor latent_image_ids = prepare_latent_image_ids(
        batch_size, adjusted_height / 2, adjusted_width / 2);
    return {packed_latents, latent_image_ids};
  }

  std::vector<torch::Tensor> forward_(
      std::optional<std::vector<std::string>> prompt = std::nullopt,
      std::optional<std::vector<std::string>> prompt_2 = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt = std::nullopt,
      std::optional<std::vector<std::string>> negative_prompt_2 = std::nullopt,
      float true_cfg_scale = 1.0f,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt,
      int64_t num_inference_steps = 28,
      float guidance_scale = 3.5f,
      int64_t num_images_per_prompt = 1,
      std::optional<int64_t> seed = std::nullopt,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> pooled_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_pooled_prompt_embeds = std::nullopt,
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    int64_t actual_height = height.has_value()
                                ? height.value()
                                : default_sample_size_ * vae_scale_factor_;
    int64_t actual_width = width.has_value()
                               ? width.value()
                               : default_sample_size_ * vae_scale_factor_;
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0);
    }
    int64_t total_batch_size = batch_size * num_images_per_prompt;
    bool has_neg_prompt = negative_prompt.has_value() ||
                          (negative_prompt_embeds.has_value() &&
                           negative_pooled_prompt_embeds.has_value());
    bool do_true_cfg = (true_cfg_scale > 1.0f) && has_neg_prompt;
    // encode prompt
    auto [encoded_prompt_embeds, encoded_pooled_embeds, text_ids] =
        encode_prompt(prompt,
                      prompt_2,
                      prompt_embeds,
                      pooled_prompt_embeds,
                      num_images_per_prompt,
                      max_sequence_length);
    // encode negative prompt
    torch::Tensor negative_encoded_embeds, negative_pooled_embeds;
    torch::Tensor negative_text_ids;
    if (do_true_cfg) {
      auto [neg_emb, neg_pooled, neg_ids] =
          encode_prompt(negative_prompt,
                        negative_prompt_2,
                        negative_prompt_embeds,
                        negative_pooled_prompt_embeds,
                        num_images_per_prompt,
                        max_sequence_length);
      negative_encoded_embeds = neg_emb;
      negative_pooled_embeds = neg_pooled;
      negative_text_ids = neg_ids;
    }
    // prepare latent
    int64_t num_channels_latents = transformer_->in_channels() / 4;
    auto [prepared_latents, latent_image_ids] =
        prepare_latents(total_batch_size,
                        num_channels_latents,
                        actual_height,
                        actual_width,
                        seed.has_value() ? seed.value() : 42,
                        latents);
    // prepare timestep
    std::vector<float> new_sigmas;
    for (int64_t i = 0; i < num_inference_steps; ++i) {
      new_sigmas.push_back(1.0f - static_cast<float>(i) /
                                      (num_inference_steps - 1) *
                                      (1.0f - 1.0f / num_inference_steps));
    }

    int64_t image_seq_len = prepared_latents.size(1);
    float mu = calculate_shift(image_seq_len,
                               scheduler_->base_image_seq_len(),
                               scheduler_->max_image_seq_len(),
                               scheduler_->base_shift(),
                               scheduler_->max_shift());
    auto [timesteps, num_inference_steps_actual] = retrieve_timesteps(
        scheduler_, num_inference_steps, device_, new_sigmas, mu);
    int64_t num_warmup_steps =
        std::max(static_cast<int64_t>(timesteps.numel()) -
                     num_inference_steps_actual * scheduler_->order(),
                 static_cast<int64_t>(0LL));
    // prepare guidance
    torch::Tensor guidance;
    if (transformer_->guidance_embeds()) {
      torch::TensorOptions options =
          torch::dtype(torch::kFloat32).device(device_);

      guidance = torch::full(at::IntArrayRef({1}), guidance_scale, options);
      guidance = guidance.expand({prepared_latents.size(0)});
    }
    scheduler_->set_begin_index(0);
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());
    // image rotary positional embeddings outplace computation
    auto [rot_emb1, rot_emb2] =
        pos_embed_->forward_cache(text_ids,
                                  latent_image_ids,
                                  height.value() / (vae_scale_factor_ * 2),
                                  width.value() / (vae_scale_factor_ * 2));
    torch::Tensor image_rotary_emb = torch::stack({rot_emb1, rot_emb2}, 0);
    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      torch::Tensor t = timesteps[i].unsqueeze(0);
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(1000.0f);
      int64_t step_id = i + 1;
      torch::Tensor noise_pred = transformer_->forward(prepared_latents,
                                                       encoded_prompt_embeds,
                                                       encoded_pooled_embeds,
                                                       timestep,
                                                       image_rotary_emb,
                                                       guidance,
                                                       step_id);
      if (do_true_cfg) {
        torch::Tensor negative_noise_pred =
            transformer_->forward(prepared_latents,
                                  negative_encoded_embeds,
                                  negative_pooled_embeds,
                                  timestep,
                                  image_rotary_emb,
                                  guidance,
                                  step_id);
        noise_pred =
            noise_pred + (noise_pred - negative_noise_pred) * true_cfg_scale;
        negative_noise_pred.reset();
      }
      auto prev_latents = scheduler_->step(noise_pred, t, prepared_latents);
      prepared_latents = prev_latents.detach();
      std::vector<torch::Tensor> tensors = {prepared_latents, noise_pred};
      noise_pred.reset();
      prev_latents = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }
    torch::Tensor image;
    // Unpack latents
    torch::Tensor unpacked_latents = unpack_latents(
        prepared_latents, actual_height, actual_width, vae_scale_factor_);
    unpacked_latents =
        (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;
    unpacked_latents = unpacked_latents.to(dtype_);
    image = vae_->decode(unpacked_latents);
    image = vae_image_processor_->postprocess(image, "pil");
    return std::vector<torch::Tensor>{{image}};
  }

 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  VAE vae_{nullptr};
  VAEImageProcessor vae_image_processor_{nullptr};
  FluxDiTModel transformer_{nullptr};
  float vae_scaling_factor_;
  float vae_shift_factor_;
  int default_sample_size_;
  FluxPosEmbed pos_embed_{nullptr};
};
TORCH_MODULE(FluxPipeline);

REGISTER_DIT_MODEL(flux, FluxPipeline);
}  // namespace xllm
