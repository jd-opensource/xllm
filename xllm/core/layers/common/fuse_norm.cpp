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

#include "fuse_norm.h"

#include <glog/logging.h>

#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

const static std::string kLayerNormMode = "layernorm";
const static std::string kRmsNormMode = "rmsnorm";

FusedRMSNormImpl::FusedRMSNormImpl(int64_t dim,
                                   double eps,
                                   const torch::TensorOptions& options)
    : norm_dim_(dim), eps_(eps), mode_(kRmsNormMode) {
  weight_ = register_parameter("weight",
                               torch::empty({dim}, options),
                               /*requires_grad=*/false);
}

torch::Tensor FusedRMSNormImpl::forward(torch::Tensor& input) {
  auto output = torch::empty_like(input);
  return forward_output(input, output);
}

torch::Tensor FusedRMSNormImpl::forward_output(torch::Tensor& input,
                                               torch::Tensor& output) {
  auto org_shape = output.sizes().vec();
  input = input.reshape({-1, norm_dim_});
  output = output.reshape({-1, norm_dim_});

  xllm::kernel::FusedLayerNormParams fused_layernorm_params;
  fused_layernorm_params.input = input;
  fused_layernorm_params.output = output;
  fused_layernorm_params.weight = weight_;
  fused_layernorm_params.mode = mode_;
  fused_layernorm_params.eps = eps_;
  if (bias_.defined()) {
    fused_layernorm_params.beta = bias_;
  }

  xllm::kernel::fused_layernorm(fused_layernorm_params);

  output = output.view(org_shape);
  return output;
}

void FusedRMSNormImpl::load_state_dict(const StateDict& state_dict) {
  LOAD_WEIGHT(weight);
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

void FusedRMSNormImpl::set_layernorm_mode() {
  mode_ = kLayerNormMode;
  bias_ = register_parameter(
      "bias", torch::empty({norm_dim_}, weight_.options()), false);
}

}  // namespace layer
}  // namespace xllm
