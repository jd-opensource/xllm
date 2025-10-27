/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "dit_batch.h"

#include <c10/core/DeviceType.h>
#include <torch/torch.h>

#include <vector>

namespace {

bool check_tensors_valid(const std::vector<torch::Tensor>& vec) {
  CHECK(!vec.empty());

  torch::Tensor ref_tensor = vec[0];
  if (!ref_tensor.defined()) return false;

  if (vec.size() == 1) return true;

  const auto ref_shape = ref_tensor.sizes();
  for (size_t i = 1; i < vec.size(); ++i) {
    if (!vec[i].defined()) return false;

    if (vec[i].sizes() != ref_shape) {
      return false;
    }
  }

  return true;
}

}  // namespace

namespace xllm {

DiTForwardInput DiTBatch::prepare_forward_input() {
  CHECK(!request_vec_.empty());

  DiTForwardInput input;
  input.batch_size = request_vec_.size();
  input.generation_params = request_vec_[0]->state().generation_params();

  std::vector<torch::Tensor> prompt_embeds;
  std::vector<torch::Tensor> pooled_prompt_embeds;

  std::vector<torch::Tensor> negative_prompt_embeds;
  std::vector<torch::Tensor> negative_pooled_prompt_embeds;

  std::vector<torch::Tensor> images;
  std::vector<torch::Tensor> mask_images;

  std::vector<torch::Tensor> latents;
  std::vector<torch::Tensor> masked_image_latents;
  for (const auto& request : request_vec_) {
    const auto& generation_params = request->state().generation_params();
    if (input.generation_params != generation_params) {
      LOG(WARNING) << " dit generation params not equal";
    }

    const auto& input_params = request->state().input_params();
    if (!input_params.prompt.empty())
      input.prompts.emplace_back(input_params.prompt);

    if (!input_params.prompt_2.empty())
      input.prompts_2.emplace_back(input_params.prompt_2);

    if (!input_params.negative_prompt.empty())
      input.negative_prompts.emplace_back(input_params.negative_prompt);

    if (!input_params.negative_prompt_2.empty())
      input.negative_prompts_2.emplace_back(input_params.negative_prompt_2);

    prompt_embeds.emplace_back(input_params.prompt_embed);
    pooled_prompt_embeds.emplace_back(input_params.pooled_prompt_embed);

    negative_prompt_embeds.emplace_back(input_params.negative_prompt_embed);
    negative_pooled_prompt_embeds.emplace_back(
        input_params.negative_pooled_prompt_embed);

    latents.emplace_back(input_params.latent);
    masked_image_latents.emplace_back(input_params.masked_image_latent);

    images.emplace_back(input_params.image);
    mask_images.emplace_back(input_params.mask_image);
  }

  if (input.prompts.size() != request_vec_.size()) {
    input.prompts.clear();
  }

  if (input.prompts_2.size() != request_vec_.size()) {
    input.prompts_2.clear();
  }

  if (input.negative_prompts.size() != request_vec_.size()) {
    input.negative_prompts.clear();
  }

  if (input.negative_prompts_2.size() != request_vec_.size()) {
    input.negative_prompts_2.clear();
  }

  if (check_tensors_valid(images)) {
    input.images = torch::stack(images);
  }

  if (check_tensors_valid(mask_images)) {
    input.mask_images = torch::stack(mask_images);
  }

  if (check_tensors_valid(prompt_embeds)) {
    input.prompt_embeds = torch::stack(prompt_embeds);
  }

  if (check_tensors_valid(pooled_prompt_embeds)) {
    input.pooled_prompt_embeds = torch::stack(pooled_prompt_embeds);
  }

  if (check_tensors_valid(negative_prompt_embeds)) {
    input.negative_prompt_embeds = torch::stack(negative_prompt_embeds);
  }

  if (check_tensors_valid(negative_pooled_prompt_embeds)) {
    input.negative_pooled_prompt_embeds =
        torch::stack(negative_pooled_prompt_embeds);
  }

  if (check_tensors_valid(latents)) {
    input.latents = torch::stack(latents);
  }

  if (check_tensors_valid(masked_image_latents)) {
    input.masked_image_latents = torch::stack(masked_image_latents);
  }
  return input;
}

void DiTBatch::process_forward_output(const DiTForwardOutput& output) {
  CHECK(request_vec_.size() == output.tensors.size());
  for (int idx = 0; idx < request_vec_.size(); ++idx) {
    auto& request = request_vec_[idx];
    request->handle_forward_output(output.tensors[idx]);
  }
}

}  // namespace xllm
