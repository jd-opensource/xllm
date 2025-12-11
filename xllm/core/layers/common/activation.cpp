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

#include "activation.h"

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

ActivationImpl::ActivationImpl(const std::string& act_mode, bool is_gated)
    : act_mode_(act_mode), is_gated_(is_gated) {}

void ActivationImpl::forward(torch::Tensor& input, torch::Tensor& output) {
  xllm::kernel::ActivationParams activation_params;
  activation_params.input = input;
  activation_params.output = output;
  activation_params.act_mode = act_mode_;
  activation_params.is_gated = is_gated_;
  xllm::kernel::active(activation_params);
}

}  // namespace layer
}  // namespace xllm