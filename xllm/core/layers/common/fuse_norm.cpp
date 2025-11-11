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
    : norm_dim_(dim), eps_(eps) {
  weight_ = register_parameter("weight",
                               torch::empty({dim}, options),
                               /*requires_grad=*/false);
}

torch::Tensor FusedRMSNormImpl::forward(torch::Tensor& input) {
  auto org_shape = input.sizes().vec();
  input = input.reshape({-1, norm_dim_});
#if defined(USE_NPU)
  xllm::kernel::FusedLayerNormParams fused_layernorm_params;
  fused_layernorm_params.input = input;
  fused_layernorm_params.weight = weight_;
  fused_layernorm_params.eps = eps_;
  fused_layernorm_params.mode = kRmsNormMode;
  auto output = xllm::kernel::fused_layernorm_tensor(fused_layernorm_params);
#else
  auto output = torch::empty_like(input);

  xllm::kernel::FusedLayerNormParams fused_layernorm_params;
  fused_layernorm_params.input = input;
  fused_layernorm_params.output = output;
  fused_layernorm_params.weight = weight_;
  fused_layernorm_params.mode = kRmsNormMode;
  fused_layernorm_params.eps = eps_;

  xllm::kernel::fused_layernorm(fused_layernorm_params);
#endif

  output = output.view(org_shape);
  return output;
}

void FusedRMSNormImpl::load_state_dict(const StateDict& state_dict) {
  const auto weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    CHECK_EQ(weight_.sizes(), weight.sizes())
        << "weight size mismatch for " << name();
    weight_.copy_(weight);
  }
}

}  // namespace layer
}  // namespace xllm
