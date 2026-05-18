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
#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include "models/dit/flowmatch_euler_discrete_scheduler.h"
namespace xllm::dit {

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
    xllm::FlowMatchEulerDiscreteScheduler scheduler,
    int64_t num_inference_steps = 0,
    torch::Device device = torch::kCPU,
    std::optional<std::vector<float>> sigmas = std::nullopt,
    std::optional<float> mu = std::nullopt) {
  torch::Tensor scheduler_timesteps;
  int64_t steps;
  if (sigmas.has_value()) {
    steps = sigmas->size();
    scheduler->set_timesteps(
        static_cast<int64_t>(steps), device, *sigmas, mu, std::nullopt);

    scheduler_timesteps = scheduler->timesteps();
  } else {
    steps = num_inference_steps;
    scheduler->set_timesteps(
        static_cast<int64_t>(steps), device, std::nullopt, mu, std::nullopt);
    scheduler_timesteps = scheduler->timesteps();
  }
  if (scheduler_timesteps.device() != device) {
    scheduler_timesteps = scheduler_timesteps.to(device);
  }
  return {scheduler_timesteps, steps};
}

std::pair<int64_t, int64_t> calculate_dimensions(double target_area,
                                                 double ratio) {
  double width = std::sqrt(target_area * ratio);
  double height = width / ratio;

  width = std::round(width / 32) * 32;
  height = std::round(height / 32) * 32;

  return {static_cast<int64_t>(width), static_cast<int64_t>(height)};
}

torch::Tensor randn_tensor(const std::vector<int64_t>& shape,
                           int64_t seed,
                           torch::TensorOptions& options) {
  if (shape.empty()) {
    LOG(FATAL) << "Shape must not be empty.";
  }
  at::Generator gen = at::detail::createCPUGenerator();
  gen = gen.clone();
  gen.set_current_seed(seed);
  torch::Tensor latents;
  latents = torch::randn(shape, gen, options.device(torch::kCPU));
  latents = latents.to(options);
  return latents;
}

class VAEImageProcessorImpl : public torch::nn::Module {
 public:
  explicit VAEImageProcessorImpl(
      ModelContext context,
      bool do_resize = true,
      bool do_normalize = true,
      bool do_binarize = false,
      bool do_convert_rgb = false,
      bool do_convert_grayscale = false,
      int64_t latent_channels = 4,
      std::optional<int64_t> scale_factor = std::nullopt) {
    const auto& model_args = context.get_model_args();
    dtype_ = context.get_tensor_options().dtype().toScalarType();
    scale_factor_ = scale_factor.has_value()
                        ? scale_factor.value()
                        : 1 << model_args.block_out_channels().size();
    latent_channels_ = latent_channels;
    do_resize_ = do_resize;
    do_normalize_ = do_normalize;
    do_binarize_ = do_binarize;
    do_convert_rgb_ = do_convert_rgb;
    do_convert_grayscale_ = do_convert_grayscale;
  }

  std::pair<int64_t, int64_t> adjust_dimensions(int64_t height,
                                                int64_t width) const {
    height = height - (height % scale_factor_);
    width = width - (width % scale_factor_);
    return {height, width};
  }

  torch::Tensor preprocess(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt,
      const std::string& resize_mode = "default",
      std::optional<std::tuple<int64_t, int64_t, int64_t, int64_t>>
          crop_coords = std::nullopt) {
    torch::Tensor processed = image.clone();
    if (processed.dtype() != torch::kFloat32) {
      processed = processed.to(torch::kFloat32);
    }
    if (processed.max().item<float>() > 1.1f) {
      processed = processed / 255.0f;
    }
    if (crop_coords.has_value()) {
      auto [x1, y1, x2, y2] = crop_coords.value();
      x1 = std::max(int64_t(0), x1);
      y1 = std::max(int64_t(0), y1);
      x2 = std::min(processed.size(-1), x2);
      y2 = std::min(processed.size(-2), y2);

      if (processed.dim() == 3) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      } else if (processed.dim() == 4) {
        processed = processed.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(),
                                     torch::indexing::Slice(y1, y2),
                                     torch::indexing::Slice(x1, x2)});
      }
    }
    int64_t channel = processed.size(1);
    if (channel == latent_channels_) {
      return image;
    }
    auto [target_h, target_w] =
        get_default_height_width(processed, height, width);
    if (do_resize_) {
      if (resize_mode == "lanczo") {
        processed = lanczo_resize(processed, target_h, target_w);
      } else if (resize_mode == "default") {
        processed = resize(processed,
                           {target_h, target_w},
                           /*resample=*/3,  // BICUBIC (approximate LANCZOS)
                           /*antialias=*/true);
      } else {
        LOG(FATAL) << "Currently only support two resize methods, 'lanczo' and "
                      "'default'"
                   << ", but got: " << resize_mode;
      }
    }

    if (do_normalize_) {
      processed = normalize(processed);
    }
    if (do_binarize_) {
      processed = (processed >= 0.5f).to(torch::kFloat32);
    }
    processed = processed.to(dtype_);
    return processed;
  }

  torch::Tensor postprocess(
      const torch::Tensor& tensor,
      const std::string& output_type = "pt",
      std::optional<std::vector<bool>> do_denormalize = std::nullopt) {
    torch::Tensor processed = tensor.clone();
    if (do_normalize_) {
      if (!do_denormalize.has_value()) {
        processed = denormalize(processed);
      } else {
        for (int64_t i = 0; i < processed.size(0); ++i) {
          if (i < do_denormalize.value().size() && do_denormalize.value()[i]) {
            processed[i] = denormalize(processed[i]);
          }
        }
      }
    }
    if (output_type == "np") {
      return processed.permute({0, 2, 3, 1}).contiguous();
    }
    return processed;
  }

 private:
  std::pair<int64_t, int64_t> get_default_height_width(
      const torch::Tensor& image,
      std::optional<int64_t> height = std::nullopt,
      std::optional<int64_t> width = std::nullopt) const {
    int64_t h, w;
    if (image.dim() == 3) {
      h = image.size(1);
      w = image.size(2);
    } else if (image.dim() == 4) {
      h = image.size(2);
      w = image.size(3);
    } else {
      LOG(FATAL) << "Unsupported image dimension: " << image.dim();
    }

    int64_t target_h = height.value_or(h);
    int64_t target_w = width.value_or(w);
    return adjust_dimensions(target_h, target_w);
  }

  torch::Tensor normalize(const torch::Tensor& tensor) const {
    return 2.0 * tensor - 1.0;
  }

  torch::Tensor denormalize(const torch::Tensor& tensor) const {
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0);
  }

 public:
  torch::Tensor resize(const torch::Tensor& image,
                       const std::vector<int64_t>& size,
                       size_t resample,
                       bool antialias) {
    if (image.dim() != 4) {
      LOG(FATAL) << "Input image must be a 4D tensor (B x C x H x W).";
    }
    auto options = torch::nn::functional::InterpolateFuncOptions()
                       .size(size)
                       .align_corners(false)
                       .antialias(antialias);
    switch (resample) {
      case 1:
        options.mode(torch::kNearest);
        break;
      case 2:
        options.mode(torch::kBilinear);
        break;
      case 3:
        options.mode(torch::kBicubic);
        break;
      default:
        LOG(FATAL) << "Invalid resample value. Must be one of 1, 2, or 3.";
    }
    return torch::nn::functional::interpolate(image, options);
  }

  // This function is used to align with the PIL Image.resize function
  // currently, the diffusers uses LANCZO mode to resize the pil image inputs,
  // but there is not a implementation for pytorch/libtorch, use
  // torch::nn::functional::Interpolate for torch tensor may cause a precision
  // problem, diffusers repo also have the same problem when using torch tensor
  // as input. to keep the same with PIL Image, we borrow the lanczo function
  // from opencv library
  torch::Tensor lanczo_resize(torch::Tensor image,
                              int64_t target_height,
                              int64_t target_width) {
    auto options = image.options();
    image = image.cpu().to(torch::kFloat32);

    bool is_batch = (image.dim() == 4);
    int64_t c = is_batch ? image.size(1) : image.size(0);
    int64_t h = is_batch ? image.size(2) : image.size(1);
    int64_t w = is_batch ? image.size(3) : image.size(2);

    cv::Mat mat(h, w, CV_32FC(c));
    float* data = image.data_ptr<float>();
    memcpy(mat.data, data, c * h * w * sizeof(float));

    cv::Mat resized;
    cv::resize(mat,
               resized,
               cv::Size(target_width, target_height),
               0,
               0,
               cv::INTER_LANCZOS4);

    torch::Tensor result =
        torch::empty({c, target_height, target_width}, image.options());
    memcpy(result.data_ptr<float>(),
           resized.data,
           c * target_height * target_width * sizeof(float));
    result = result.to(options);

    return is_batch ? result.unsqueeze(0) : result;
  }

 private:
  int64_t scale_factor_ = 8;
  int64_t latent_channels_ = 4;
  bool do_resize_ = true;
  bool do_normalize_ = true;
  bool do_binarize_ = false;
  bool do_convert_rgb_ = false;
  bool do_convert_grayscale_ = false;
  torch::ScalarType dtype_ = torch::kFloat32;
};
TORCH_MODULE(VAEImageProcessor);

}  // namespace xllm::dit
