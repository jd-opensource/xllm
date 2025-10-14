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

#include "framework/model/model_args.h"
#include "framework/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/linear.h"

namespace xllm {
namespace layer {

class Qwen3MLPImpl : public torch::nn::Module {
 public:
  Qwen3MLPImpl() = default;
  Qwen3MLPImpl(const ModelArgs& args,
               const QuantArgs& quant_args,
               const ParallelArgs& parallel_args,
               const torch::TensorOptions& options);

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& residual);

  void load_state_dict(const StateDict& state_dict);

 private:
  bool is_gated_;
  int64_t intermediate_size_;
  ParallelArgs parallel_args_;
  ColumnParallelLinear gate_up_proj_{nullptr};
  RowParallelLinear down_proj_{nullptr};
};
TORCH_MODULE(Qwen3MLP);

}  // namespace layer
}  // namespace xllm