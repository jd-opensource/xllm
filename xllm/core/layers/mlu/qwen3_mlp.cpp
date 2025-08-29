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

#include "qwen3_mlp.h"

#include <glog/logging.h>

namespace xllm {

Qwen3MLP::Qwen3MLP(const ModelArgs& args,
                   const QuantArgs& quant_args,
                   const ParallelArgs& parallel_args,
                   const torch::TensorOptions& options) {
  // 1. gate + up
  gate_up_proj_ =
      register_module("gate_up_proj",
                      ColumnParallelLinear(args.hidden_size(),
                                           args.intermediate_size() * 2,
                                           /*bias=*/false,
                                           /*gather_output=*/false,
                                           quant_args,
                                           parallel_args,
                                           options));

  // 2. down
  down_proj_ = register_module("down_proj",
                               RowParallelLinear(args.intermediate_size(),
                                                 args.hidden_size(),
                                                 /*bias=*/false,
                                                 /*input_is_parallelized=*/true,
                                                 /*if_reduce_results=*/true,
                                                 quant_args,
                                                 parallel_args,
                                                 options));
}

torch::Tensor Qwen3MLP::forward(const torch::Tensor& hidden_states) {
  auto gate_up = gate_up_proj_->forward(hidden_states);
  auto chunks = gate_up.chunk(2, /*dim=*/-1);
  auto gate = torch::silu(chunks[0]);
  auto up = chunks[1];
  return down_proj_->forward(gate * up);
}

void Qwen3MLP::load_state_dict(const StateDict& state_dict) {
  gate_up_proj_->load_state_dict(
      state_dict.get_dict_with_prefix("gate_up_proj."));
  down_proj_->load_state_dict(state_dict.get_dict_with_prefix("down_proj."));
}

}  // namespace xllm