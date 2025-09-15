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

#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/npu/pos_embedding.h"
#include "core/layers/npu/rms_norm.h"
#include "core/layers/npu/word_embedding.h"
#include "core/layers/rotary_embedding.h"
#include "framework/model_context.h"
#include "models/autoencoder_kl.h"
#include "models/clip_text_model.h"
#include "models/flux/dit.h"
#include "models/flux/flowmatch_euler_discrete_scheduler.h"
#include "models/model_registry.h"
#include "models/t5_encoder.h"
namespace xllm::hf {
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
    std::optional<int64_t> num_inference_steps = std::nullopt,
    std::optional<torch::Device> device = std::nullopt,
    std::optional<std::vector<int64_t>> timesteps = std::nullopt,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  if (timesteps.has_value() && sigmas.has_value()) {
    throw std::invalid_argument(
        "Only one of `timesteps` or `sigmas` can be provided. Please choose "
        "one.");
  }
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (timesteps.has_value()) {
    std::vector<float> timesteps_float;
    timesteps_float.reserve(timesteps->size());
    for (int64_t val : *timesteps) {
      timesteps_float.push_back(static_cast<float>(val));
    }
    steps = timesteps->size();
    scheduler->set_timesteps(static_cast<int>(steps),
                             device.has_value() ? *device : torch::kCPU,
                             std::nullopt,
                             mu,
                             timesteps_float);

    scheduler_timesteps = scheduler->timesteps();
  } else if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(static_cast<int>(steps),
                             device.has_value() ? *device : torch::kCPU,
                             *sigmas,
                             mu,
                             std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    if (!num_inference_steps.has_value()) {
      throw std::invalid_argument(
          "Either `num_inference_steps`, `timesteps` or `sigmas` must be "
          "provided.");
    }
    steps = *num_inference_steps;
    scheduler->set_timesteps(static_cast<int>(steps),
                             device.has_value() ? *device : torch::kCPU,
                             std::nullopt,
                             mu,
                             std::nullopt);
    scheduler_timesteps = scheduler->timesteps();
  }
  if (device.has_value() && scheduler_timesteps.device() != *device) {
    scheduler_timesteps = scheduler_timesteps.to(*device);
  }
  return {scheduler_timesteps, steps};
}
torch::Tensor randn_tensor(
    const std::vector<int64_t>& shape,
    const std::optional<torch::Generator>& generator = std::nullopt,
    torch::Device device = torch::kCPU,
    torch::ScalarType dtype = torch::kFloat32,
    torch::Layout layout = torch::kStrided) {
  if (shape.empty()) {
    throw std::invalid_argument("Shape cannot be empty.");
  }
  torch::Device rand_device = device;
  bool generator_provided = false;  // generator.has_value();

  // if (generator_provided) {
  //     LOG(INFO) << "Generator provided, device: " << device;

  //     torch::DeviceType gen_device_type = generator.value().device().type();
  //     LOG(INFO) << "Generator device type: " << gen_device_type << ", target
  //     device type: " << device.type();
  //     if (gen_device_type != device.type() && gen_device_type == torch::kCPU)
  //     {
  //       LOG(INFO)
  //           << "Generator is on CPU, but target device is " << device
  //           << ". Generating on CPU and moving to target device.";
  //         rand_device = torch::kCPU;
  //         if (device.type() != torch::kMPS) {
  //             LOG(INFO)
  //                 << "The passed generator was created on 'cpu' even though a
  //                 tensor on "
  //                 << device << " was expected. Tensors will be created on
  //                 'cpu' and then moved to "
  //                 << device << ".";
  //         }
  //     }
  //     else if (gen_device_type != device.type() && gen_device_type !=
  //     torch::kCPU) {
  //         LOG(ERROR)
  //             << "Generator device type (" << gen_device_type
  //             << ") does not match target device type (" << device.type() <<
  //             ").";
  //         throw std::invalid_argument(
  //             "Cannot generate a " + device.str() + " tensor from a generator
  //             of type " + torch::Device(gen_device_type).str() + "."
  //         );
  //     }
  // } else {
  //     LOG(INFO) << "No generator provided, using default, device: " <<
  //     device;
  // }

  // if (generator_provided && !generator.value().defined()) {
  //     throw std::invalid_argument("Provided generator is not defined.");
  // }
  LOG(INFO) << "begin to generate random tensor with shape: " << shape
            << ", device: " << rand_device << ", dtype: " << dtype;
  torch::manual_seed(42);
  torch::Tensor latents;
  torch::TensorOptions options =
      torch::dtype(dtype).device(rand_device).layout(layout);

  try {
    if (generator_provided) {
      LOG(INFO) << "Generating random latents with shape: " << shape
                << ", device: " << rand_device << ", dtype: " << dtype;
      latents = torch::randn(shape, generator.value(), options);
    } else {
      LOG(INFO) << "Generating random latents with shape: " << shape
                << ", device: " << rand_device << ", dtype: " << dtype;
      latents = torch::randn(shape, options);
    }

    if (latents.device() != device) {
      latents = latents.to(device);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error generating random tensor: " << e.what();
    throw;
  }

  return latents;
}

struct FluxPipelineOutput {
  std::vector<torch::Tensor> images;
};
class FluxPipelineImpl : public torch::nn::Module {
 private:
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
  VAE vae_{nullptr};
  VAEImageProcessor vae_image_processor_{nullptr};
  DiTModelPipeline transformer_{nullptr};
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
  ModelArgs model_args_;
  torch::TensorOptions options_;

 public:
  FluxPipelineImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    vae_scale_factor_ = 1 << (model_args_.vae_block_out_channels().size() - 1);
    _execution_device = options_.device();
    _execution_dtype = torch::kBFloat16;
    LOG(INFO) << _execution_device << " is the execution device";
    LOG(INFO) << _execution_dtype << " is the execution dtype";
    LOG(INFO) << model_args_;
    vae_shift_factor_ = model_args_.vae_shift_factor();
    vae_scaling_factor_ = model_args_.vae_scale_factor();
    default_sample_size_ = 128;
    tokenizer_max_length_ = 77;  // TODO: get from config file
    LOG(INFO) << "Initializing Flux pipeline...";
    vae_image_processor_ = VAEImageProcessor(
        true, vae_scale_factor_, 4, "lanczos", -1, true, false, false, false);
    LOG(INFO) << "VAE image processor initialized.";
    vae_ = VAE(context, _execution_device, _execution_dtype);
    LOG(INFO) << "VAE initialized.";
    transformer_ =
        DiTModelPipeline(context, _execution_device, _execution_dtype);
    LOG(INFO) << "DiT transformer initialized.";
    t5_ = T5EncoderModel(context, _execution_device, _execution_dtype);
    LOG(INFO) << "T5 initialized.";
    clip_text_model_ =
        CLIPTextModel(context, _execution_device, _execution_dtype);
    LOG(INFO) << "CLIP text model initialized.";
    scheduler_ = FlowMatchEulerDiscreteScheduler(context);
    LOG(INFO) << "Flux pipeline initialized.";
    // register modules
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
  void check_inputs(
      std::optional<torch::optional<std::vector<std::string>>> prompt,
      std::optional<torch::optional<std::vector<std::string>>> prompt_2,
      int64_t height,
      int64_t width,
      std::optional<torch::optional<std::vector<std::string>>> negative_prompt,
      std::optional<torch::optional<std::vector<std::string>>>
          negative_prompt_2,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> negative_prompt_embeds,
      std::optional<torch::Tensor> pooled_prompt_embeds,
      std::optional<torch::Tensor> pooled_negative_prompt_embeds,
      std::optional<int64_t> max_sequence_length) {
    const int64_t divisor = vae_scale_factor_ * 2;
    if (height % divisor != 0 || width % divisor != 0) {
      LOG(WARNING) << "`height` and `width` have to be divisible by " << divisor
                   << " but are " << height << " and " << width
                   << ". Dimensions will be resized accordingly";
    }

    if (prompt.has_value() && prompt_embeds.has_value()) {
      throw std::invalid_argument(
          "Cannot forward both `prompt` and `prompt_embeds`. Please make sure "
          "to only forward one of the two.");
    }
    if (prompt_2.has_value() && prompt_embeds.has_value()) {
      throw std::invalid_argument(
          "Cannot forward both `prompt_2` and `prompt_embeds`. Please make "
          "sure to only forward one of the two.");
    }
    if (!prompt.has_value() && !prompt_embeds.has_value()) {
      throw std::invalid_argument(
          "Provide either `prompt` or `prompt_embeds`. Cannot leave both "
          "undefined.");
    }

    if (negative_prompt.has_value() && negative_prompt_embeds.has_value()) {
      throw std::invalid_argument(
          "Cannot forward both `negative_prompt` and `negative_prompt_embeds`. "
          "Please make sure to only forward one of the two.");
    }
    if (negative_prompt_2.has_value() && negative_prompt_embeds.has_value()) {
      throw std::invalid_argument(
          "Cannot forward both `negative_prompt_2` and "
          "`negative_prompt_embeds`. Please make sure to only forward one of "
          "the two.");
    }

    if (prompt_embeds.has_value() && !pooled_prompt_embeds.has_value()) {
      throw std::invalid_argument(
          "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have "
          "to be passed. "
          "Make sure to generate `pooled_prompt_embeds` from the same text "
          "encoder that was used for `prompt_embeds`.");
    }

    if (negative_prompt_embeds.has_value() &&
        !pooled_negative_prompt_embeds.has_value()) {
      throw std::invalid_argument(
          "If `negative_prompt_embeds` are provided, "
          "`pooled_negative_prompt_embeds` also have to be passed. "
          "Make sure to generate `pooled_negative_prompt_embeds` from the same "
          "text encoder that was used for `negative_prompt_embeds`.");
    }

    if (max_sequence_length.has_value() && max_sequence_length.value() > 512) {
      throw std::invalid_argument(
          "`max_sequence_length` cannot be greater than 512 but is " +
          std::to_string(max_sequence_length.value()));
    }
  }
  torch::Tensor _prepare_latent_image_ids(int64_t batch_size,
                                          int64_t height,
                                          int64_t width,
                                          torch::Device device,
                                          torch::ScalarType dtype) {
    torch::Tensor latent_image_ids =
        torch::zeros({height, width, 3}, torch::dtype(dtype).device(device));
    torch::Tensor height_range =
        torch::arange(height, torch::dtype(dtype).device(device)).unsqueeze(1);
    latent_image_ids.select(2, 1) += height_range;
    torch::Tensor width_range =
        torch::arange(width, torch::dtype(dtype).device(device)).unsqueeze(0);
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
  std::pair<torch::Tensor, torch::Tensor> prepare_latents(
      int64_t batch_size,
      int64_t num_channels_latents,
      int64_t height,
      int64_t width,
      torch::ScalarType dtype,
      torch::Device device,
      torch::Generator generator,
      std::optional<torch::Tensor> latents = std::nullopt) {
    int64_t adjusted_height = 2 * (height / (vae_scale_factor_ * 2));
    int64_t adjusted_width = 2 * (width / (vae_scale_factor_ * 2));
    std::vector<int64_t> shape = {
        batch_size, num_channels_latents, adjusted_height, adjusted_width};
    if (latents.has_value()) {
      torch::Tensor latent_image_ids = _prepare_latent_image_ids(
          batch_size, adjusted_height / 2, adjusted_width / 2, device, dtype);
      return {latents.value().to(device).to(dtype), latent_image_ids};
    }
    torch::manual_seed(42);
    torch::Tensor latents_tensor =
        randn_tensor(shape, generator, device, dtype);
    torch::Tensor packed_latents = _pack_latents(latents_tensor,
                                                 batch_size,
                                                 num_channels_latents,
                                                 adjusted_height,
                                                 adjusted_width);
    torch::Tensor latent_image_ids = _prepare_latent_image_ids(
        batch_size, adjusted_height / 2, adjusted_width / 2, device, dtype);
    return {packed_latents, latent_image_ids};
  }
  torch::Tensor forward(const InputParams& input_params,
                        const GenerationParams& generation_params) {
    LOG(INFO) << "FluxPipelineImpl forward called" << input_params.prompt;
    torch::Generator generator = torch::Generator();
    torch::manual_seed(generation_params.seed.value_or(42));
    std::vector<torch::Generator> generators_vec;
    generators_vec.push_back(generator);
    std::optional<std::vector<torch::Generator>> generators_opt =
        generators_vec;
    FluxPipelineOutput output = forward_(
        std::make_optional(c10::optional<std::vector<std::string>>{
            std::vector<std::string>{input_params.prompt}}),  // prompt
        std::make_optional(
            c10::optional<std::vector<std::string>>{std::vector<std::string>{
                input_params.prompt_2.value_or("")}}),  // prompt_2
        std::make_optional(c10::optional<std::vector<std::string>>{
            std::vector<std::string>{input_params.negative_prompt.value_or(
                "")}}),  // negative_prompt
        std::make_optional(c10::optional<std::vector<std::string>>{
            std::vector<std::string>{input_params.negative_prompt_2.value_or(
                "")}}),                                // negative_prompt_2
        generation_params.true_cfg_scale.value_or(1),  // cfg scale
        std::make_optional(generation_params.height),  // height
        std::make_optional(generation_params.width),   // width
        generation_params.num_inference_steps.value_or(
            28),                                         // num_inference_steps
        std::nullopt,                                    // sigmas
        generation_params.guidance_scale.value_or(3.5),  // guidance_scale
        generation_params.num_images_per_prompt.value_or(
            1),                               // num_images_per_prompt
        generators_opt,                       // generator
        input_params.latents,                 // latents
        input_params.prompt_embeds,           // prompt_embeds
        input_params.negative_prompt_embeds,  // negative_prompt_embeds
        input_params.pooled_prompt_embeds,    // pooled_prompt_embeds
        input_params
            .negative_pooled_prompt_embeds,  // negative_pooled_prompt_embeds
        "pil",                               // output_type
        generation_params.max_sequence_length.value_or(
            512)  // max_sequence_length
    );
    return output.images[0];
  }
  torch::Tensor _get_clip_prompt_embeds(
      std::vector<std::string>& prompt,
      std::optional<torch::Device> device = std::nullopt,
      int64_t num_images_per_prompt = 1) {
    torch::Device used_device =
        device.has_value() ? device.value() : _execution_device;
    std::vector<std::string> prompt_list = prompt;
    int64_t batch_size = prompt_list.size();
    TORCH_CHECK(batch_size > 0, "Prompt list cannot be empty");
    std::vector<std::string> processed_prompt = prompt_list;
    // TODO add CLIP tokenizer
    //    auto text_inputs = tokenizer.encode(
    //        processed_prompt,
    //        tokenizer_max_length_,
    //        true,
    //        true
    //    );
    //    torch::Tensor text_input_ids = text_inputs.input_ids;
    //    auto untruncated = tokenizer.encode(
    //        processed_prompt,
    //        0,
    //        true,
    //        false
    //    );
    //    torch::Tensor untruncated_ids = untruncated.input_ids;
    //    if (untruncated_ids.size(1) >= text_input_ids.size(1) &&
    //        !torch::equal(
    //            text_input_ids,
    //            untruncated_ids.index({torch::indexing::Slice(),
    //            torch::indexing::Slice(0, text_input_ids.size(1))})
    //        )) {
    //
    //        auto truncated_part = untruncated_ids.index({
    //            torch::indexing::Slice(),
    //            torch::indexing::Slice(tokenizer_max_length_ - 1, -1)
    //        });
    //        auto removed_text = tokenizer.batch_decode(truncated_part);

    //       std::cerr << "Warning: The following part of your input was
    //       truncated because CLIP can only handle sequences up to "
    //                 << tokenizer_max_length_ << " tokens: ";
    //       for (const auto& text : removed_text) {
    //           std::cerr << text << " ";
    //       }
    //       std::cerr << std::endl;
    //   }
    std::vector<int64_t> text_input_ids = {
        49406, 40555, 3155,  1844,  267,   12177, 2463,  268,   8893,  6469,
        268,   1844,  1611,  49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407};
    auto encoder_output = clip_text_model_->forward(text_input_ids);
    torch::Tensor prompt_embeds = encoder_output;
    prompt_embeds = prompt_embeds.to(used_device).to(_execution_dtype);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, -1});
    return prompt_embeds;
  }

  torch::Tensor _get_t5_prompt_embeds(
      std::vector<std::string>& prompt,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512,
      std::optional<torch::Device> device = std::nullopt,
      std::optional<torch::Dtype> dtype = std::nullopt) {
    torch::Device used_device =
        device.has_value() ? device.value() : _execution_device;
    torch::Dtype used_dtype =
        dtype.has_value() ? dtype.value() : _execution_dtype;
    std::vector<std::string> prompt_list = prompt;
    int64_t batch_size = prompt_list.size();
    TORCH_CHECK(batch_size > 0, "Prompt list cannot be empty");
    std::vector<std::string> processed_prompt = prompt_list;
    // TODO add T5 tokenizer
    //    auto text_inputs = tokenizer_2.encode(
    //        processed_prompt,
    //        max_sequence_length,
    //        true,
    //        true
    //    );
    //  torch::Tensor text_input_ids = text_inputs.input_ids;
    torch::Tensor text_input_ids = t5_->create_text_ids();
    //   auto untruncated = tokenizer_2.encode(
    //       processed_prompt,
    //       0,
    //       true,
    //       false
    //   );
    //   torch::Tensor untruncated_ids = untruncated.input_ids;
    //   if (untruncated_ids.size(1) >= text_input_ids.size(1) &&
    //       !torch::equal(text_input_ids,
    //       untruncated_ids.index({torch::indexing::Slice(),
    //       torch::indexing::Slice(0, text_input_ids.size(1))}))) { auto
    //       truncated_part = untruncated_ids.index({
    //           torch::indexing::Slice(),
    //           torch::indexing::Slice(max_sequence_length - 1, -1)
    //       });
    //       auto removed_text = tokenizer_2.batch_decode(truncated_part);

    //       std::cerr << "Warning: The following part of your input was
    //       truncated because `max_sequence_length` is set to "
    //                 << max_sequence_length << " tokens: ";
    //       for (const auto& text : removed_text) {
    //           std::cerr << text << " ";
    //       }
    //       std::cerr << std::endl;
    //   }
    torch::Tensor prompt_embeds = t5_->forward(text_input_ids.to(used_device));
    prompt_embeds = prompt_embeds.to(used_dtype).to(used_device);
    int64_t seq_len = prompt_embeds.size(1);
    prompt_embeds = prompt_embeds.repeat({1, num_images_per_prompt, 1});
    prompt_embeds =
        prompt_embeds.view({batch_size * num_images_per_prompt, seq_len, -1});
    return prompt_embeds;
  }
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> encode_prompt(
      std::optional<torch::optional<std::vector<std::string>>> prompt,
      std::optional<torch::optional<std::vector<std::string>>> prompt_2,
      std::optional<torch::Tensor> prompt_embeds,
      std::optional<torch::Tensor> pooled_prompt_embeds,
      std::optional<torch::Device> device,
      int64_t num_images_per_prompt = 1,
      int64_t max_sequence_length = 512) {
    torch::Device used_device =
        device.has_value() ? device.value() : _execution_device;
    std::vector<std::string> prompt_list;
    if (prompt.has_value()) {
      auto inner_prompt = prompt.value();
      if (inner_prompt.has_value()) {
        prompt_list = inner_prompt.value();
      }
    }
    if (prompt_list.empty()) {
      prompt_list = {""};
    }
    if (!prompt_embeds.has_value()) {
      std::vector<std::string> prompt_2_list;
      if (prompt_2.has_value()) {
        auto inner_prompt2 = prompt_2.value();
        if (inner_prompt2.has_value()) {
          prompt_2_list = inner_prompt2.value();
        }
      }
      if (prompt_2_list.empty()) {
        prompt_2_list = prompt_list;
      }

      pooled_prompt_embeds = _get_clip_prompt_embeds(
          prompt_list, used_device, num_images_per_prompt);
      prompt_embeds = _get_t5_prompt_embeds(prompt_2_list,
                                            num_images_per_prompt,
                                            max_sequence_length,
                                            used_device,
                                            std::nullopt);
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
      std::optional<torch::optional<std::vector<std::string>>> prompt =
          std::nullopt,
      std::optional<torch::optional<std::vector<std::string>>> prompt_2 =
          std::nullopt,
      std::optional<torch::optional<std::vector<std::string>>> negative_prompt =
          std::nullopt,
      std::optional<torch::optional<std::vector<std::string>>>
          negative_prompt_2 = std::nullopt,
      float true_cfg_scale = 1.0f,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt,
      int64_t num_inference_steps = 28,
      std::optional<std::vector<float>> sigmas = std::nullopt,
      float guidance_scale = 3.5f,
      int64_t num_images_per_prompt = 1,
      std::optional<std::vector<torch::Generator>> generators = std::nullopt,
      std::optional<torch::Tensor> latents = std::nullopt,
      std::optional<torch::Tensor> prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> pooled_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_prompt_embeds = std::nullopt,
      std::optional<torch::Tensor> negative_pooled_prompt_embeds = std::nullopt,
      std::string output_type = "pil",
      int64_t max_sequence_length = 512) {
    LOG(INFO) << "FluxPipeline operator() called";
    torch::NoGradGuard no_grad;
    int64_t actual_height = height.has_value()
                                ? height.value()
                                : default_sample_size_ * vae_scale_factor_;
    int64_t actual_width = width.has_value()
                               ? width.value()
                               : default_sample_size_ * vae_scale_factor_;
    // check inputs
    //   check_inputs(
    //       prompt, prompt_2, actual_height, actual_width,
    //       negative_prompt, negative_prompt_2,
    //       prompt_embeds, negative_prompt_embeds,
    //       pooled_prompt_embeds, negative_pooled_prompt_embeds,
    //       max_sequence_length
    //   );
    _guidance_scale = guidance_scale;
    _current_timestep = std::nullopt;
    _interrupt = false;
    int64_t batch_size;
    if (prompt.has_value()) {
      if (prompt.value().has_value()) {
        batch_size = prompt.value().value().size();
      } else {
        batch_size = prompt_embeds.value().size(0);
      }
    } else {
      batch_size = prompt_embeds.value().size(0);
    }
    int64_t total_batch_size = 1;  // batch_size * num_images_per_prompt;
    torch::Device device = _execution_device;
    bool has_neg_prompt = negative_prompt.has_value() ||
                          (negative_prompt_embeds.has_value() &&
                           negative_pooled_prompt_embeds.has_value());
    bool do_true_cfg = (true_cfg_scale > 1.0f) && has_neg_prompt;
    batch_size = 1;
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
    auto [prepared_latents, latent_image_ids] = prepare_latents(
        total_batch_size,
        num_channels_latents,
        actual_height,
        actual_width,
        _execution_dtype,
        device,
        generators.has_value() ? generators.value()[0] : torch::Generator(),
        latents);
    // prepare timestep
    std::vector<float> new_sigmas;
    if (!sigmas.has_value()) {
      for (int64_t i = 0; i < num_inference_steps; ++i) {
        new_sigmas.push_back(1.0f - static_cast<float>(i) /
                                        (num_inference_steps - 1) *
                                        (1.0f - 1.0f / num_inference_steps));
      }
    } else {
      new_sigmas = sigmas.value();
    }
    int64_t image_seq_len = prepared_latents.size(1);
    float mu = calculate_shift(image_seq_len,
                               scheduler_->base_image_seq_len(),
                               scheduler_->max_image_seq_len(),
                               scheduler_->base_shift(),
                               scheduler_->max_shift());
    auto [timesteps, num_inference_steps_actual] = retrieve_timesteps(
        scheduler_, num_inference_steps, device, std::nullopt, new_sigmas, mu);
    int64_t num_warmup_steps =
        std::max(static_cast<int64_t>(timesteps.numel()) -
                     num_inference_steps_actual * scheduler_->order,
                 static_cast<int64_t>(0LL));
    _num_timesteps = timesteps.numel();
    // prepare guidance
    torch::Tensor guidance;
    if (transformer_->guidance_embeds()) {
      LOG(INFO) << "Preparing guidance tensor with scale: " << guidance_scale;
      torch::TensorOptions options =
          torch::dtype(torch::kFloat32).device(device);

      guidance = torch::full(at::IntArrayRef({1}), guidance_scale, options);
      guidance = guidance.expand({prepared_latents.size(0)});
    }
    scheduler_->set_begin_index(0);
    LOG(INFO) << "Starting inference with " << num_inference_steps_actual
              << " steps.";
    torch::Tensor timestep =
        torch::empty({prepared_latents.size(0)}, prepared_latents.options());
    for (int64_t i = 0; i < timesteps.numel(); ++i) {
      if (_interrupt) break;

      torch::Tensor t = timesteps[i].unsqueeze(0);
      _current_timestep = t;
      timestep.fill_(t.item<float>())
          .to(prepared_latents.dtype())
          .div_(1000.0f);
      torch::Tensor noise_pred = transformer_->forward(prepared_latents,
                                                       encoded_prompt_embeds,
                                                       encoded_pooled_embeds,
                                                       timestep,
                                                       latent_image_ids,
                                                       text_ids,
                                                       guidance,
                                                       0);
      if (do_true_cfg) {
        torch::Tensor negative_noise_pred =
            transformer_->forward(prepared_latents,
                                  negative_encoded_embeds,
                                  negative_pooled_embeds,
                                  timestep,
                                  latent_image_ids,
                                  negative_text_ids,
                                  guidance,
                                  0);
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
      image = vae_->decode(unpacked_latents).sample;
      image = vae_image_processor_->postprocess(image, output_type);
    }
    auto bytes = torch::pickle_save(image.cpu());  // 转成二进制 pickle 数据
    std::ofstream fout(
        "/export/home/liuyiming54/precision_test/flux/xllm/final_image.pkl",
        std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
    return FluxPipelineOutput{{image}};
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "FluxPipeline loading model...";
    LOG(INFO) << "Loading Flux model from: " << loader->model_root_path()
              << loader->component_names();
    std::string model_path = loader->model_root_path();
    auto transformer_loader =
        loader->take_sub_model_loader_by_folder("transformer");
    auto vae_loader = loader->take_sub_model_loader_by_folder("vae");
    auto t5_loader = loader->take_sub_model_loader_by_folder("text_encoder_2");
    auto clip_loader = loader->take_sub_model_loader_by_folder("text_encoder");
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
  }
};
TORCH_MODULE(FluxPipeline);
REGISTER_DIT_MODEL(flux, FluxPipeline);
}  // namespace xllm::hf