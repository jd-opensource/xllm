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

#include <string>

namespace xllm {
namespace layer {

class ActivationImpl : public torch::nn::Module {
 public:
  ActivationImpl(const std::string& act_mode, bool is_gated);

  void forward(torch::Tensor& input, torch::Tensor& output);

 private:
  std::string act_mode_;
  bool is_gated_;
};
TORCH_MODULE(Activation);

}  // namespace layer
}  // namespace xllm