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

#include "qwen3_moe_mlp.h"

#include <glog/logging.h>

#include "kernels/mlu/torch_ops_api.h"

namespace xllm {
namespace layer {

Qwen3MoeMLPImpl::Qwen3MoeMLPImpl(const ModelArgs& args,
                                 const QuantArgs& quant_args,
                                 const ParallelArgs& parallel_args,
                                 const torch::TensorOptions& options) {
  // gate
  gate_ = register_module(
      "gate_proj",
      ReplicatedLinear(
          args.hidden_size(), args.num_experts(), false, quant_args, options));
  // expert
  expert_ = register_module("experts",
                            FusedMoE(args.num_experts(),
                                     args.num_experts_per_tok(),
                                     args.hidden_size(),
                                     args.moe_intermediate_size(),
                                     true,
                                     false,
                                     false,
                                     args.norm_topk_prob(),
                                     args.hidden_act(),
                                     "softmax",
                                     quant_args,
                                     parallel_args,
                                     options));
}

torch::Tensor Qwen3MoeMLPImpl::forward(const torch::Tensor& hidden_states,
                                       c10::optional<torch::Tensor> residual) {
  auto router_logits = gate_(hidden_states, c10::nullopt);
  return expert_(hidden_states, router_logits, residual);
}

void Qwen3MoeMLPImpl::load_state_dict(const StateDict& state_dict) {
  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  expert_->load_state_dict(state_dict.get_dict_with_prefix("experts."));
}

}  // namespace layer
}  // namespace xllm