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

#include "atb/atb_infer.h"
#include "buffer/atb_workspace.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm::hf {

class LlmHeadImpl : public torch::nn::Module {
 public:
  ~LlmHeadImpl() override = default;

  virtual void load_state_dict(const StateDict& state_dict) = 0;

  virtual void verify_loaded_weights(const std::string weight_str) const = 0;

  virtual void merge_loaded_weights() = 0;

  virtual torch::Tensor forward(const torch::Tensor& hidden_states,
                                const torch::Tensor& seleted_idxes,
                                int nodeId) = 0;
};

class LlmHead : public torch::nn::ModuleHolder<LlmHeadImpl> {
 public:
  using torch::nn::ModuleHolder<LlmHeadImpl>::ModuleHolder;
  using Impl __attribute__((__unused__)) = LlmHeadImpl;

  LlmHead(const ModelContext& context);
};

std::shared_ptr<LlmHeadImpl> create_llm_head_layer(const ModelContext& context);

}  // namespace xllm::hf
