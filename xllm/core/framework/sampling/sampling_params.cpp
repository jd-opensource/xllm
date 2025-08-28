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

#include "sampling_params.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace xllm {

void SamplingParameters::init(
    const std::vector<const RequestSamplingParam*>& req_sampling_params,
    const std::vector<int32_t>& selected_token_idxes,
    const std::vector<int32_t>& sample_idxes,
    const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
    const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
    const std::vector<int32_t>& unique_token_lens_vec) {
  CHECK_EQ(req_sampling_params.size(), selected_token_idxes.size());
  CHECK_GE(req_sampling_params.size(), sample_idxes.size());
  CHECK_EQ(req_sampling_params.size(), unique_token_ids_vec.size());
  CHECK_EQ(req_sampling_params.size(), unique_token_counts_vec.size());
  CHECK_EQ(req_sampling_params.size(), unique_token_lens_vec.size());

  std::vector<float> frequency_penalties;
  std::vector<float> presence_penalties;
  std::vector<float> repetition_penalties;
  std::vector<float> temperatures;
  std::vector<float> top_p;
  std::vector<int64_t> top_k;
  bool logprobs = false;
  int64_t max_top_logprobs = 0;
  bool is_embeddings = false;
  for (const auto* p : req_sampling_params) {
    frequency_penalties.push_back(p->frequency_penalty);
    presence_penalties.push_back(p->presence_penalty);
    repetition_penalties.push_back(p->repetition_penalty);
    temperatures.push_back(p->temperature);
    top_p.push_back(p->top_p);
    top_k.push_back(p->top_k);
    logprobs = logprobs || p->logprobs;
    is_embeddings = is_embeddings || p->is_embeddings;
    max_top_logprobs = std::max(max_top_logprobs, p->top_logprobs);
  }

  bool need_token_stats = false;

  // Create tensor on cpu pinned memory here
  auto int_tensor_options = torch::TensorOptions()
                                .device(torch::kCPU)
                                .dtype(torch::kInt)
                                .pinned_memory(true);
  auto int64_tensor_options = torch::TensorOptions()
                                  .device(torch::kCPU)
                                  .dtype(torch::kInt64)
                                  .pinned_memory(true);
  auto float32_tensor_options = torch::TensorOptions()
                                    .device(torch::kCPU)
                                    .dtype(torch::kFloat32)
                                    .pinned_memory(true);
  auto bool_tensor_options = torch::TensorOptions()
                                 .device(torch::kCPU)
                                 .dtype(torch::kBool)
                                 .pinned_memory(true);
  if (std::any_of(frequency_penalties.begin(),
                  frequency_penalties.end(),
                  [](float t) { return t != 0.0; }) ||
      std::any_of(presence_penalties.begin(),
                  presence_penalties.end(),
                  [](float t) { return t != 0.0; })) {
    this->frequency_penalties =
        torch::tensor(frequency_penalties, float32_tensor_options);
    this->presence_penalties =
        torch::tensor(presence_penalties, float32_tensor_options);
    need_token_stats = true;
  }
  if (std::any_of(repetition_penalties.begin(),
                  repetition_penalties.end(),
                  [](float t) { return t != 1.0; })) {
    this->repetition_penalties =
        torch::tensor(repetition_penalties, float32_tensor_options);
    need_token_stats = true;
  }
  if (std::any_of(temperatures.begin(), temperatures.end(), [](float t) {
        return t != 0.0 && t != 1.0;
      })) {
    this->temperatures = torch::tensor(temperatures, float32_tensor_options);
  }
  if (std::any_of(
          top_k.begin(), top_k.end(), [](int64_t t) { return t > 0; })) {
    this->top_k = torch::tensor(top_k, int64_tensor_options);
  }
  if (std::any_of(
          top_p.begin(), top_p.end(), [](float t) { return t != 1.0; })) {
    this->top_p = torch::tensor(top_p, float32_tensor_options);
  }

  this->selected_token_idxes =
      torch::tensor(selected_token_idxes, int_tensor_options);
  if (need_token_stats) {
    this->unique_token_ids =
        create_2d_tensor(unique_token_ids_vec, torch::kInt64);
    this->unique_token_counts =
        create_2d_tensor(unique_token_counts_vec, torch::kInt);
    this->unique_token_ids_lens =
        torch::tensor(unique_token_lens_vec, int_tensor_options);
  }

  // construct do sample tensor
  std::vector<int32_t> do_sample;
  for (const auto idx : sample_idxes) {
    const auto* p = req_sampling_params[idx];
    // need to do sample if any of following is true
    const bool sample = p->do_sample || p->temperature != 0.0 ||
                        p->top_p != 1.0 || p->top_k > 0;
    do_sample.push_back(sample ? 1 : 0);
  }
  this->sample_idxes = torch::tensor(sample_idxes, int_tensor_options);
  this->do_sample = torch::tensor(do_sample, bool_tensor_options);
  this->logprobs = logprobs;
  this->max_top_logprobs = max_top_logprobs;
  this->is_embeddings = is_embeddings;
}

}  // namespace xllm
