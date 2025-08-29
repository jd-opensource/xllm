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

#include <ATen/core/TensorBody.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace detail {

inline torch::Tensor norm(const torch::Tensor& x, float eps) {
  const auto mean = x.pow(/*exponent=*/2).mean(/*dim=*/-1, /*keepdim=*/true);
  return x * torch::rsqrt(mean + eps);
}

inline torch::Tensor rms_norm(const torch::Tensor& input,
                              const torch::Tensor& weight,
                              float eps) {
  // it is important to use float to calculate the mean and std
  const auto output = norm(input.to(torch::kFloat), eps);
  // convert back to the original dtype
  return output.to(input) * weight;
}

inline torch::Tensor gemma_rms_norm(const torch::Tensor& input,
                                    const torch::Tensor& weight,
                                    float eps) {
  // it is important to use float to calculate the mean and std
  auto output = norm(input.to(torch::kFloat), eps);
  // Llama does x.to(float16) * w whilst Gemma is (x * (w + 1)).to(float16)
  // See https://github.com/huggingface/transformers/pull/29402
  output = output * (1.0 + weight.to(torch::kFloat));
  return output.type_as(input);
}

inline torch::Tensor rms_norm_residual(const torch::Tensor& input,
                                       torch::Tensor& residual,
                                       const torch::Tensor& weight,
                                       float eps) {
  // it is important to use float for the residual
  auto x = input.to(torch::kFloat) + residual.to(torch::kFloat);
  residual = x.to(input);
  const auto output = norm(x, eps);
  // convert back to the original dtype
  return output.to(input) * weight;
}

inline torch::Tensor layer_norm(torch::Tensor input,
                                const std::vector<int64_t>& normalized_shape,
                                const torch::Tensor& weight,
                                const torch::Tensor& bias,
                                double eps) {
  namespace F = torch::nn::functional;
  return F::detail::layer_norm(input, normalized_shape, weight, bias, eps);
}

}  // namespace detail

// apply layer normalization over a mini-batch of inputs as described in
// the paper `Layer Normalization`: https://arxiv.org/abs/1607.06450
// x = ((x - mean(x)) / sqrt(std(x) + eps)) * weight + bias
class LayerNormImpl : public torch::nn::Module {
 public:
  // dim: the dim over which the mean and std are calculated separately.
  // eps: a value added to the denominator for numerical stability.
  LayerNormImpl(int64_t dim,
                float eps,
                bool bias,
                const torch::TensorOptions& options)
      : eps_(eps) {
    normalized_shape_ = {dim};
    weight_ = register_parameter("weight",
                                 torch::empty(normalized_shape_, options),
                                 /*requires_grad=*/false);
    if (bias) {
      bias_ = register_parameter("bias",
                                 torch::zeros(normalized_shape_, options),
                                 /*requires_grad=*/false);
    }
  }

  torch::Tensor forward(torch::Tensor input) {
    namespace F = torch::nn::functional;
    return F::detail::layer_norm(
        input, normalized_shape_, weight_, bias_, eps_);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      weight_is_loaded_ = true;
    }
    if (bias_.defined()) {
      const auto bias = state_dict.get_tensor("bias");
      if (bias.defined()) {
        CHECK_EQ(bias_.sizes(), bias.sizes())
            << "bias size mismatch for " << name();
        bias_.copy_(bias);
        bias_is_loaded_ = true;
      }
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const {
    CHECK(weight_is_loaded_)
        << "weight is not loaded for " << prefix + "weight";
    CHECK(!bias_.defined() || bias_is_loaded_)
        << "bias is not loaded for " << prefix + "bias";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  torch::Tensor bias_{nullptr};

  // whether the weight is loaded
  bool weight_is_loaded_ = false;
  bool bias_is_loaded_ = false;

  // configs
  float eps_;
  std::vector<int64_t> normalized_shape_;
};
TORCH_MODULE(LayerNorm);

// Root mean square normalization
class RMSNormImpl : public torch::nn::Module {
 public:
  RMSNormImpl(int64_t dim, float eps, const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight",
                                 torch::empty({dim}, options),
                                 /*requires_grad=*/false);
  }

  torch::Tensor forward(const torch::Tensor& input) {
    return detail::rms_norm(input, weight_, eps_);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const {
    CHECK(is_loaded_) << "weight is not loaded for " << prefix + "weight";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // configs
  float eps_;
};
TORCH_MODULE(RMSNorm);

class GemmaRMSNormImpl : public torch::nn::Module {
 public:
  GemmaRMSNormImpl(int64_t dim, float eps, const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight",
                                 torch::empty({dim}, options),
                                 /*requires_grad=*/false);
  }

  torch::Tensor forward(const torch::Tensor& input) {
    return detail::gemma_rms_norm(input, weight_, eps_);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const {
    CHECK(is_loaded_) << "weight is not loaded for " << prefix + "weight";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // configs
  float eps_;
};
TORCH_MODULE(GemmaRMSNorm);

// Root mean square normalization
class RMSNormResidualImpl : public torch::nn::Module {
 public:
  RMSNormResidualImpl(int64_t dim,
                      float eps,
                      const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight",
                                 torch::empty({dim}, options),
                                 /*requires_grad=*/false);
  }

  torch::Tensor forward(const torch::Tensor& input, torch::Tensor& residual) {
    if (residual.defined()) {
      return detail::rms_norm_residual(input, residual, weight_, eps_);
    }
    residual = input;
    return detail::rms_norm(input, weight_, eps_);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      CHECK_EQ(weight_.sizes(), weight.sizes())
          << "weight size mismatch for " << name();
      weight_.copy_(weight);
      is_loaded_ = true;
    }
  }

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const {
    CHECK(is_loaded_) << "weight is not loaded for " << prefix + "weight";
  }

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // configs
  float eps_;
};
TORCH_MODULE(RMSNormResidual);

}  // namespace xllm
