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
#include <torch/types.h>

#include <memory>

#include "framework/model/model_args.h"

namespace xllm {
namespace layer {

class RotaryEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryEmbeddingImpl(int rotary_dim,
                      int max_position_embeddings,
                      int rope_theta,
                      bool interleaved,
                      const torch::TensorOptions& options);

  void forward(torch::Tensor& q,
               torch::Tensor& k,
               const torch::Tensor& positions,
               const torch::Tensor& cu_query_lens,
               int max_query_len,
               bool is_prompt);

  torch::Tensor get_cos_sin_cache() { return cos_sin_cache_; }

 private:
  int rotary_dim_;
  int max_position_embeddings_;
  int rope_theta_;
  bool interleaved_;
  torch::Tensor sin_;
  torch::Tensor cos_;
  torch::Tensor cos_sin_cache_;
};
TORCH_MODULE(RotaryEmbedding);

}  // namespace layer
}  // namespace xllm
