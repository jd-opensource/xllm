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

#include "core/framework/state_dict/utils.h"
namespace xllm {
namespace F = torch::nn::functional;

class DiTLinearImpl : public torch::nn::Module {
 public:
  DiTLinearImpl(int64_t in, int64_t out, bool with_bias = true) {
    weight = register_parameter("weight", torch::empty({out, in}));
    if (with_bias) {
      bias = register_parameter("bias", torch::empty(out));
    } else {
      bias = register_parameter("bias", {}, false);
    }
  }

  torch::Tensor forward(const torch::Tensor& x) {
    return F::linear(x, weight, bias);
  }

  void load_state_dict(const StateDict& state_dict) {
    weight::load_weight(state_dict, "weight", weight, weight_is_loaded_);
    if (bias.defined()) {
      weight::load_weight(state_dict, "bias", bias, bias_is_loaded_);
    }
  }

  void to(torch::TensorOptions options) {
    weight.set_data(weight.to(options));
    if (bias.defined()) {
      bias.set_data(bias.to(options));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    if (bias.defined()) {
      CHECK(bias_is_loaded_) << "bias is not loaded for " << prefix + "bias";
    }
  }

  torch::Tensor weight;
  torch::Tensor bias;

 private:
  bool weight_is_loaded_{false};
  bool bias_is_loaded_{false};
};

TORCH_MODULE(DiTLinear);
}  // namespace xllm
