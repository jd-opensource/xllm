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

#include "framework/context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {

class MultiheadAttentionImpl : public torch::nn::Module {
 public:
  MultiheadAttentionImpl(const Context& context);

  torch::Tensor forward(torch::Tensor query,
                        torch::Tensor key,
                        torch::Tensor value,
                        torch::Tensor key_padding_mask);

  void load_state_dict(const StateDict& state_dict);

  void verify_loaded_weights(const std::string& prefix) const;

 private:
  int64_t n_head_;
  int64_t head_dim_;
  int64_t hidden_size_;
  torch::TensorOptions options_;

  torch::Tensor in_proj_weight_;
  torch::Tensor in_proj_bias_;
  torch::Tensor out_proj_weight_;
  torch::Tensor out_proj_bias_;

  bool is_in_proj_weight_loaded_;
  bool is_in_proj_bias_loaded_;
  bool is_out_proj_weight_loaded_;
  bool is_out_proj_bias_loaded_;
};

TORCH_MODULE(MultiheadAttention);

}  // namespace xllm