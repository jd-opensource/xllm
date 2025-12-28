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
#include <torch/torch.h>

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "dit_request_output.h"

namespace xllm {

using DiTOutputFunc = std::function<bool(const DiTRequestOutput& output)>;
using DiTOutputsFunc = std::function<std::vector<bool>(
    const std::vector<DiTRequestOutput>& outputs)>;

class Call;

struct DiTGenerationParams {
  bool operator==(const DiTGenerationParams& other) const {
    return width == other.width && height == other.height &&
           num_inference_steps == other.num_inference_steps &&
           true_cfg_scale == other.true_cfg_scale &&
           guidance_scale == other.guidance_scale &&
           num_images_per_prompt == other.num_images_per_prompt &&
           seed == other.seed &&
           max_sequence_length == other.max_sequence_length &&
           strength == other.strength &&
           enable_cfg_renorm == other.enable_cfg_renorm &&
           cfg_renorm_min == other.cfg_renorm_min;
  }

  bool operator!=(const DiTGenerationParams& other) const {
    return !(*this == other);
  }

  int32_t width = 512;

  int32_t height = 512;

  int32_t num_inference_steps = 28;

  float true_cfg_scale = 1.0;

  float guidance_scale = 3.5;

  uint32_t num_images_per_prompt = 1;

  int64_t seed = 0;

  int32_t max_sequence_length = 512;

  float strength = 1.0;

  bool enable_cfg_renorm = true;

  float cfg_renorm_min = 0.0f;
};

struct DiTInputParams {
  // Primary input text description for image generation
  std::string prompt;

  // Secondary prompt for additional details (e.g., color, lighting)
  std::string prompt_2;

  // Negative prompt to exclude low-quality features
  std::string negative_prompt;

  // Secondary negative prompt to exclude additional unwanted features
  std::string negative_prompt_2;

  torch::Tensor prompt_embed;

  torch::Tensor pooled_prompt_embed;

  torch::Tensor negative_prompt_embed;

  torch::Tensor negative_pooled_prompt_embed;

  torch::Tensor latent;

  torch::Tensor image;

  torch::Tensor control_image;

  torch::Tensor mask_image;

  torch::Tensor masked_image_latent;
};

struct DiTRequestState {
 public:
  DiTRequestState(DiTInputParams& input_params,
                  DiTGenerationParams& generation_params,
                  const DiTOutputFunc& output_func,
                  const DiTOutputsFunc& outputs_func,
                  std::optional<Call*> call = std::nullopt)
      : input_params_(std::move(input_params)),
        generation_params_(std::move(generation_params)),
        output_func_(std::move(output_func)),
        outputs_func_(std::move(outputs_func)),
        call_(call) {}
  DiTRequestState() {}
  DiTInputParams& input_params() { return input_params_; }
  DiTGenerationParams& generation_params() { return generation_params_; }
  DiTOutputFunc& output_func() { return output_func_; }
  DiTOutputsFunc& outputs_func() { return outputs_func_; }
  std::optional<Call*>& call() { return call_; }

 private:
  DiTInputParams input_params_;
  DiTGenerationParams generation_params_;
  DiTOutputFunc output_func_;
  DiTOutputsFunc outputs_func_;
  std::optional<Call*> call_;
};

}  // namespace xllm
