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
  int32_t width = 512;

  int32_t height = 512;

  std::optional<int32_t> num_inference_steps;

  std::optional<float> true_cfg_scale;

  std::optional<float> guidance_scale;

  std::optional<uint32_t> num_images_per_prompt = 1;

  std::optional<int64_t> seed;

  std::optional<int32_t> max_sequence_length;
};

struct DiTInputParams {
  std::string prompt;

  std::optional<std::string> prompt_2;

  std::optional<std::string> negative_prompt;

  std::optional<std::string> negative_prompt_2;

  // std::optional<std::string> ip_adapter_image;

  // std::optional<std::string> negative_ip_adapter_image;

  std::optional<torch::Tensor> prompt_embeds;

  std::optional<torch::Tensor> pooled_prompt_embeds;

  // std::optional<std::vector<std::vector<std::vector<float>>>>
  //     ip_adapter_image_embeds;

  std::optional<torch::Tensor> negative_prompt_embeds;

  std::optional<torch::Tensor> negative_pooled_prompt_embeds;

  // std::optional<std::vector<std::vector<std::vector<float>>>>
  //     negative_ip_adapter_image_embeds;

  std::optional<torch::Tensor> latents;

  DiTInputParams to(torch::Device device, torch::ScalarType dtype) const {
    DiTInputParams copy = *this;
    if (copy.prompt_embeds) {
      copy.prompt_embeds = copy.prompt_embeds->to(device, dtype);
    }
    if (copy.pooled_prompt_embeds) {
      copy.pooled_prompt_embeds = copy.pooled_prompt_embeds->to(device, dtype);
    }
    if (copy.negative_prompt_embeds) {
      copy.negative_prompt_embeds =
          copy.negative_prompt_embeds->to(device, dtype);
    }
    if (copy.negative_pooled_prompt_embeds) {
      copy.negative_pooled_prompt_embeds =
          copy.negative_pooled_prompt_embeds->to(device, dtype);
    }
    if (copy.latents) {
      copy.latents = copy.latents->to(device, dtype);
    }
    return copy;
  }
};

struct DiTRequestState {
 public:
  DiTRequestState(DiTInputParams& input_params,
                  DiTGenerationParams& generation_params,
                  const DiTOutputFunc& output_func,
                  const DiTOutputsFunc& outputs_func,
                  std::optional<Call*> call = std::nullopt)
      : input_params_(input_params),
        generation_params_(generation_params),
        output_func_(output_func),
        outputs_func_(outputs_func),
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
