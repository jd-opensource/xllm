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
#include <acl/acl.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <string>

#include "autoencoder_kl.h"
#include "autoencoder_kl_qwenimage.h"
#include "clip_text_model.h"
#include "core/framework/dit_model_loader.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context.h"
#include "core/framework/request/dit_request_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/framework/state_dict/utils.h"
#include "dit_qwen_image.h"
#include "flowmatch_euler_discrete_scheduler.h"
#include "models/model_registry.h"
#include "models/vlm/npu/qwen2_5_vl.h"
#include "processors/qwen2_vl_image_processor.h"
#include "t5_encoder.h"
namespace xllm {
namespace qwenimage {

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

class QwenImagePipelineBaseImpl : public torch::nn::Module {
 protected:
  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_;
  torch::TensorOptions options_;
  AutoencoderKLQwenImage vae_{nullptr};
  VAEImageProcessor vae_image_processor_{nullptr};
  std::unique_ptr<Qwen2VLImageProcessor> qwen_image_processor_{nullptr};
  Qwen2_5_VLForConditionalGeneration text_encoder{nullptr};
  QwenImageTransformer2DModel transformer_{nullptr};
  std::unique_ptr<Tokenizer> qwen_tokenizer_;
  std::unique_ptr<Tokenizer> tokenizer_;
  FlowMatchEulerDiscreteScheduler scheduler_{nullptr};
};
}  // namespace qwenimage
}  // namespace xllm
