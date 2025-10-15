#pragma once
#include <acl/acl.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "autoencoder_kl.h"
#include "clip_text_model.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/pos_embedding.h"
#include "core/layers/rms_norm.h"
#include "core/layers/rotary_embedding.h"
#include "dit.h"
#include "flowmatch_euler_discrete_scheduler.h"
#include "framework/model_context.h"
#include "models/model_registry.h"
#include "t5_encoder.h"
namespace xllm {

float calculate_shift(int64_t image_seq_len,
                      int64_t base_seq_len = 256,
                      int64_t max_seq_len = 4096,
                      float base_shift = 0.5f,
                      float max_shift = 1.15f) {
  float m =
      (max_shift - base_shift) / static_cast<float>(max_seq_len - base_seq_len);
  float b = base_shift - m * static_cast<float>(base_seq_len);
  float mu = static_cast<float>(image_seq_len) * m + b;
  return mu;
}

std::pair<torch::Tensor, int64_t> retrieve_timesteps(
    FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(
        static_cast<int>(steps), device, std::nullopt, mu, std::nullopt);
    scheduler_timesteps = scheduler->timesteps();
  }
  if (scheduler_timesteps.device() != device) {
    scheduler_timesteps = scheduler_timesteps.to(device);
  }
  return {scheduler_timesteps, steps};
}

torch::Tensor randn_tensor(const std::vector<int64_t>& shape,
                           int64_t seed,
                           torch::TensorOptions& options) {
  if (shape.empty()) {
    throw std::invalid_argument("Shape cannot be empty.");
  }
  at::Generator gen = at::detail::createCPUGenerator();
  gen = gen.clone();
  gen.set_current_seed(seed);
  torch::Tensor latents;
  try {
    latents = torch::randn(shape, gen, options.device(torch::kCPU));
    latents = latents.to(options);

  } catch (const std::exception& e) {
    LOG(ERROR) << "Error generating random tensor: " << e.what();
    throw;
  }

  return latents;
}

inline torch::Tensor get_1d_rotary_pos_embed(
    int64_t dim,
    const torch::Tensor& pos,
    float theta = 10000.0,
    bool use_real = false,
    float linear_factor = 1.0,
    float ntk_factor = 1.0,
    bool repeat_interleave_real = true,
    torch::Dtype freqs_dtype = torch::kFloat32) {
  TORCH_CHECK(dim % 2 == 0, "Dimension must be even");

  torch::Tensor pos_tensor = pos;
  if (pos.dim() == 0) {
    pos_tensor = torch::arange(pos.item<int64_t>(), pos.options());
  }

  theta = theta * ntk_factor;

  auto freqs =
      1.0 /
      (torch::pow(
           theta,
           torch::arange(
               0, dim, 2, torch::dtype(freqs_dtype).device(pos.device())) /
               dim) *
       linear_factor);  // [D/2]

  auto tensors = {pos_tensor, freqs};

  auto freqs_outer = torch::einsum("s,d->sd", tensors);  // [S, D/2]
#if defined(USE_NPU)
  freqs_outer = freqs_outer.to(torch::kFloat32);
#endif
  if (use_real && repeat_interleave_real) {
    auto cos_vals = torch::cos(freqs_outer);  // [S, D/2]
    auto sin_vals = torch::sin(freqs_outer);  // [S, D/2]

    auto freqs_cos = cos_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]

    auto freqs_sin = sin_vals.transpose(-1, -2)
                         .repeat_interleave(2, -2)
                         .transpose(-1, -2)
                         .to(torch::kFloat32);  // [S, D]
    return torch::cat({freqs_cos.unsqueeze(0), freqs_sin.unsqueeze(0)},
                      0);  // [2, S, D]
  }
}

class FluxPosEmbedImpl : public torch::nn::Module {
 public:
  FluxPosEmbedImpl(int64_t theta, std::vector<int64_t> axes_dim) {
    theta_ = theta;
    axes_dim_ = axes_dim;
  }

  std::pair<torch::Tensor, torch::Tensor> forward_cache(
      const torch::Tensor& txt_ids,
      const torch::Tensor& img_ids,
      int64_t height = -1,
      int64_t width = -1) {
    auto seq_len = txt_ids.size(0);

    // recompute the cache if height or width changes
    if (height != cached_image_height_ || width != cached_image_width_ ||
        seq_len != max_seq_len_) {
      torch::Tensor ids = torch::cat({txt_ids, img_ids}, 0);
      cached_image_height_ = height;
      cached_image_width_ = width;
      max_seq_len_ = seq_len;
      auto [cos, sin] = forward(ids);
      freqs_cos_cache_ = std::move(cos);
      freqs_sin_cache_ = std::move(sin);
    }
    return {freqs_cos_cache_, freqs_sin_cache_};
  }

  std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& ids) {
    int64_t n_axes = ids.size(-1);
    std::vector<torch::Tensor> cos_out, sin_out;
    auto pos = ids.to(torch::kFloat32);
    bool is_mps = (ids.device().type() == torch::kMPS);
    torch::Dtype freqs_dtype = is_mps ? torch::kFloat32 : torch::kFloat64;
    for (int64_t i = 0; i < n_axes; ++i) {
      auto pos_slice = pos.select(-1, i);
      auto result = get_1d_rotary_pos_embed(axes_dim_[i],
                                            pos_slice,
                                            theta_,
                                            true,  // repeat_interleave_real
                                            1,
                                            1,
                                            true,  // use_real
                                            freqs_dtype);
      auto cos = result[0];
      auto sin = result[1];
      cos_out.push_back(cos);
      sin_out.push_back(sin);
    }

    auto freqs_cos = torch::cat(cos_out, -1);
    auto freqs_sin = torch::cat(sin_out, -1);

    return {freqs_cos, freqs_sin};
  }

 private:
  int64_t theta_;
  std::vector<int64_t> axes_dim_;
  torch::Tensor freqs_cos_cache_;
  torch::Tensor freqs_sin_cache_;
  int64_t max_seq_len_ = -1;
  int64_t cached_image_height_ = -1;
  int64_t cached_image_width_ = -1;
};
TORCH_MODULE(FluxPosEmbed);

struct FluxPipelineOutput {
  std::vector<torch::Tensor> images;
};

class FluxPipelineImpl : public torch::nn::Module {
 public:
  explicit FluxPipelineImpl(const DiTModelContext& context)
      : options_(context.get_tensor_options()) {
    const auto& model_args = context.get_model_args("vae");
    vae_scale_factor_ = 1 << (model_args.vae_block_out_channels().size() - 1);
    _execution_device = options_.device();
    _execution_dtype = options_.dtype().toScalarType();

    vae_shift_factor_ = model_args.vae_shift_factor();
    vae_scaling_factor_ = model_args.vae_scale_factor();
    default_sample_size_ = 128;
    tokenizer_max_length_ = 77;  // TODO: get from config file
    LOG(INFO) << "Initializing Flux pipeline...";
    vae_image_processor_ =
        VAEImageProcessor(true, vae_scale_factor_, true, false, false, false);
    vae_ = VAE(context.get_model_context("vae"));
    LOG(INFO) << "VAE initialized.";
    pos_embed_ = register_module(
        "pos_embed",
        FluxPosEmbed(
            10000, context.get_model_args("transformer").dit_axes_dims_rope()));
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

    FluxPipelineOutput output = forward_(
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
        "pil",                                 // output_type
        generation_params.max_sequence_length  // max_sequence_length
    );

    DiTForwardOutput out;
    out.tensors = torch::chunk(output.images[0], output.images[0].size(0), 0);
    LOG(INFO) << "Output tensor chunks size: " << out.tensors.size();
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "FluxPipeline loading model from" << loader->model_root_path();
    // transformer_.to(options_);
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
    transformer_->to(_execution_device);
    vae_->load_model(std::move(vae_loader));
    vae_->to(_execution_device);
    t5_->load_model(std::move(t5_loader));
    t5_->to(_execution_device);
    clip_text_model_->load_model(std::move(clip_loader));
    clip_text_model_->to(_execution_device);
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
      torch::Tensor latent_image_ids = _prepare_latent_image_ids(
          batch_size, adjusted_height / 2, adjusted_width / 2);
      return {latents.value(), latent_image_ids};
    }
    torch::Tensor latents_tensor = randn_tensor(shape, seed, options_);
    torch::Tensor packed_latents = _pack_latents(latents_tensor,
                                                 batch_size,
                                                 num_channels_latents,
                                                 adjusted_height,
                                                 adjusted_width);
    torch::Tensor latent_image_ids = _prepare_latent_image_ids(
        batch_size, adjusted_height / 2, adjusted_width / 2);
    return {packed_latents, latent_image_ids};
  }

  torch::Tensor _prepare_latent_image_ids(int64_t batch_size,
                                          int64_t height,
                                          int64_t width) {
    torch::Tensor latent_image_ids = torch::zeros({height, width, 3}, options_);
    torch::Tensor height_range = torch::arange(height, options_).unsqueeze(1);
    latent_image_ids.select(2, 1) += height_range;
    torch::Tensor width_range = torch::arange(width, options_).unsqueeze(0);
    latent_image_ids.select(2, 2) += width_range;
    latent_image_ids = latent_image_ids.view({height * width, 3});
    return latent_image_ids;
  }

  torch::Tensor _pack_latents(const torch::Tensor& latents,
                              int64_t batch_size,
                              int64_t num_channels_latents,
                              int64_t height,
                              int64_t width) {
    torch::Tensor packed = latents.view(
        {batch_size, num_channels_latents, height / 2, 2, width / 2, 2});
    packed = packed.permute({0, 2, 4, 1, 3, 5});
    packed = packed.reshape(
        {batch_size, (height / 2) * (width / 2), num_channels_latents * 4});

    return packed;
  }

  torch::Tensor _unpack_latents(const torch::Tensor& latents,
                                int64_t height,
                                int64_t width,
                                int64_t vae_scale_factor) {
    int64_t batch_size = latents.size(0);
    int64_t num_patches = latents.size(1);
    int64_t channels = latents.size(2);
    int64_t adjusted_height = 2 * (height / (vae_scale_factor * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor * 2));
    torch::Tensor unpacked = latents.view({batch_size,
                                           adjusted_height / 2,
                                           adjusted_width / 2,
                                           channels / 4,
                                           2,
                                           2});
    unpacked = unpacked.permute({0, 3, 1, 4, 2, 5});
    unpacked = unpacked.reshape(
        {batch_size, channels / (2 * 2), adjusted_height, adjusted_width});

    return unpacked;
  }

  torch::Tensor _get_clip_prompt_embeds(std::vector<std::string>& prompt,
                                        torch::Device device = torch::kCPU,
                                        int64_t num_images_per_prompt = 1) {
    torch::Device used_device = device;
    std::vector<std::string> prompt_list = prompt;
    int64_t batch_size = prompt_list.size();
    TORCH_CHECK(batch_size > 0, "Prompt list cannot be empty");
    std::vector<std::string> processed_prompt = prompt_list;

    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);
    CHECK(tokenizer_->batch_encode(processed_prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      LOG(INFO) << "CLIP Original IDs size: " << ids;
      ids.resize(tokenizer_max_length_, 49407);
      ids.back() = 49407;
    }
    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * tokenizer_max_length_);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, tokenizer_max_length_})
            .to(used_device);
    auto encoder_output = clip_text_model_->forward(input_ids);
    torch::Tensor prompt_embeds = encoder_output;
    prompt_embeds = prompt_embeds.to(used_device).to(_execution_dtype);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, -1});
    return prompt_embeds;
  }

  torch::Tensor _get_t5_prompt_embeds(std::vector<std::string>& prompt,
                                      int64_t num_images_per_prompt = 1,
                                      int64_t max_sequence_length = 512,
                                      torch::Device device = torch::kCPU) {
    torch::Device used_device = device;
    torch::Dtype used_dtype = _execution_dtype;
    std::vector<std::string> prompt_list = prompt;
    int64_t batch_size = prompt_list.size();
    TORCH_CHECK(batch_size > 0, "Prompt list cannot be empty");
    std::vector<std::string> processed_prompt = prompt_list;

    std::vector<std::vector<int32_t>> text_input_ids;
    text_input_ids.reserve(batch_size);
    CHECK(tokenizer_2_->batch_encode(processed_prompt, &text_input_ids));
    for (auto& ids : text_input_ids) {
      LOG(INFO) << "T5 Original IDs size: " << ids;
      ids.resize(max_sequence_length, 0);
    }

    std::vector<int32_t> text_input_ids_flat;
    text_input_ids_flat.reserve(batch_size * max_sequence_length);
    for (const auto& ids : text_input_ids) {
      text_input_ids_flat.insert(
          text_input_ids_flat.end(), ids.begin(), ids.end());
    }
    auto input_ids =
        torch::tensor(text_input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, max_sequence_length})
            .to(used_device);
    torch::Tensor prompt_embeds = t5_->forward(input_ids);
    prompt_embeds = prompt_embeds.to(used_dtype).to(used_device);
    int64_t seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, seq_len, -1});
    return prompt_embeds;
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<std::vector<std::string>> prompt,
      std::optional<std::vector<std::string>> prompt_2,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> pooled_prompt_embeds,
      torch::Device device,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512) {
    torch::Device used_device = device;
    std::vector<std::string> prompt_list;
    if (prompt.has_value()) {
      prompt_list = prompt.value();
    }
    if (prompt_list.empty()) {
      prompt_list = {""};
    }
    if (!prompt_embeds.has_value()) {
      std::vector<std::string> prompt_2_list;
      if (prompt_2.has_value()) {
        prompt_2_list = prompt_2.value();
      }
      if (prompt_2_list.empty()) {
        prompt_2_list = prompt_list;
      }
      pooled_prompt_embeds = _get_clip_prompt_embeds(
          prompt_list, used_device, num_images_per_prompt);
      prompt_embeds = _get_t5_prompt_embeds(prompt_2_list,
                                            num_images_per_prompt,
                                            max_sequence_length,
                                            used_device);
    }
    torch::Dtype dtype = _execution_dtype;
    torch::Tensor text_ids =
        torch::zeros({prompt_embeds.value().size(1), 3},
                     torch::device(used_device).dtype(dtype));

    return std::make_tuple(prompt_embeds.value(),
                           pooled_prompt_embeds.has_value()
                               ? pooled_prompt_embeds.value()
                               : torch::Tensor(),
                           text_ids);
  }

  FluxPipelineOutput forward_(
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
      std::string output_type = "pil",
      int64_t max_sequence_length = 512) {
    torch::NoGradGuard no_grad;
    int64_t actual_height = height.has_value()
                                ? height.value()
                                : default_sample_size_ * vae_scale_factor_;
    int64_t actual_width = width.has_value()
                               ? width.value()
                               : default_sample_size_ * vae_scale_factor_;
    _guidance_scale = guidance_scale;
    _current_timestep = std::nullopt;
    _interrupt = false;
    int64_t batch_size;
    if (prompt.has_value()) {
      batch_size = prompt.value().size();
    } else {
      batch_size = prompt_embeds.value().size(0);
    }
    int64_t total_batch_size = batch_size * num_images_per_prompt;
    torch::Device device = _execution_device;
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
                      device,
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
                        device,
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
        scheduler_, num_inference_steps, device, new_sigmas, mu);
    int64_t num_warmup_steps =
        std::max(static_cast<int64_t>(timesteps.numel()) -
                     num_inference_steps_actual * scheduler_->order(),
                 static_cast<int64_t>(0LL));
    _num_timesteps = timesteps.numel();
    // prepare guidance
    torch::Tensor guidance;
    if (transformer_->guidance_embeds()) {
      torch::TensorOptions options =
          torch::dtype(torch::kFloat32).device(device);

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
      if (_interrupt) break;

      torch::Tensor t = timesteps[i].unsqueeze(0);
      _current_timestep = t;
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
      prepared_latents = prev_latents.prev_sample.detach();
      std::vector<torch::Tensor> tensors = {prepared_latents, noise_pred};
      noise_pred.reset();
      prev_latents.prev_sample = torch::Tensor();

      if (latents.has_value() &&
          prepared_latents.dtype() != latents.value().dtype()) {
        prepared_latents = prepared_latents.to(latents.value().dtype());
      }
    }
    torch::Tensor image;
    if (output_type == "latent") {
      image = prepared_latents;
    } else {
      // Unpack latents
      torch::Tensor unpacked_latents = _unpack_latents(
          prepared_latents, actual_height, actual_width, vae_scale_factor_);
      unpacked_latents =
          (unpacked_latents / vae_scaling_factor_) + vae_shift_factor_;
      unpacked_latents = unpacked_latents.to(_execution_dtype);
      image = vae_->decode(unpacked_latents).sample;
      image = vae_image_processor_->postprocess(image, output_type);
    }
    return FluxPipelineOutput{{image}};
  }

 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  VAE vae_{nullptr};
  VAEImageProcessor vae_image_processor_{nullptr};
  FluxDiTModel transformer_{nullptr};
  T5EncoderModel t5_{nullptr};
  CLIPTextModel clip_text_model_{nullptr};
  int vae_scale_factor_;
  float vae_scaling_factor_;
  float vae_shift_factor_;
  int tokenizer_max_length_;
  int default_sample_size_;
  float _guidance_scale = 1.0f;
  std::optional<torch::Tensor> _current_timestep;
  int _num_timesteps = 0;
  bool _interrupt = false;
  torch::Device _execution_device = torch::kCPU;
  torch::ScalarType _execution_dtype = torch::kFloat32;
  torch::TensorOptions options_;
  FluxPosEmbed pos_embed_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Tokenizer> tokenizer_2_;
};
TORCH_MODULE(FluxPipeline);

REGISTER_DIT_MODEL(flux, FluxPipeline);
}  // namespace xllm
