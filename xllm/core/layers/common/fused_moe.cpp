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

#include "fused_moe.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

FusedMoEImpl::FusedMoEImpl(int64_t /*num_experts*/,
                           int64_t /*top_k*/,
                           int64_t /*num_expert_group*/,
                           int64_t /*topk_group*/,
                           double /*route_scale*/,
                           int64_t /*hidden_size*/,
                           int64_t /*intermediate_size*/,
                           int64_t /*n_shared_experts*/,
                           bool /*is_gated*/,
                           bool /*has_score_bias*/,
                           bool /*has_bias*/,
                           bool /*skip_bias_add*/,
                           int64_t /*renormalize*/,
                           const std::string& /*hidden_act*/,
                           const std::string& /*scoring_func*/,
                           const std::string& /*topk_method*/,
                           const QuantArgs& /*quant_args*/,
                           const ParallelArgs& /*parallel_args*/,
                           const torch::TensorOptions& /*options*/) {
  LOG(FATAL) << "FusedMoE is not supported for this backend. "
                "Please use MLU or ILU backend for MoE models.";
}

torch::Tensor FusedMoEImpl::forward_experts(
    const torch::Tensor& /*hidden_states*/,
    const torch::Tensor& /*router_logits*/,
    bool /*enable_all2all_communication*/) {
  LOG(FATAL) << "FusedMoE is not supported for this backend. "
                "Please use MLU or ILU backend for MoE models.";
  return torch::Tensor();
}

torch::Tensor FusedMoEImpl::forward(const torch::Tensor& /*hidden_states*/,
                                    const ModelInputParams& /*input_params*/) {
  LOG(FATAL) << "FusedMoE is not supported for this backend. "
                "Please use MLU or ILU backend for MoE models.";
  return torch::Tensor();
}

void FusedMoEImpl::load_state_dict(const StateDict& /*state_dict*/) {
  LOG(FATAL) << "FusedMoE is not supported for this backend. "
                "Please use MLU or ILU backend for MoE models.";
}

}  // namespace layer
}  // namespace xllm