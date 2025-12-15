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

#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

// for flashinfer, maybe we need to refactor later
struct StepwiseAttentionState {
  int layer_id = -1;
  torch::Tensor plan_info;
  std::string uri;

  void update(int layer_id,
              const AttentionMetadata& attn_meta,
              c10::ScalarType query_dtype,
              c10::ScalarType key_dtype,
              c10::ScalarType output_dtype,
              int head_dim_qk,
              int head_dim_vo,
              int num_qo_heads,
              int num_kv_heads,
              int block_size,
              int window_size_left,
              bool enable_cuda_graph,
              bool causal);
};

class AttentionImpl : public torch::nn::Module {
 public:
  AttentionImpl() = default;

  AttentionImpl(int layer_id,
                int num_heads,
                int head_size,
                float scale,
                int num_kv_heads,
                int sliding_window);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>> forward(
      const AttentionMetadata& attn_metadata,
      torch::Tensor& query,
      torch::Tensor& key,
      torch::Tensor& value,
      KVCache& kv_cache);

 private:
  int layer_id_;
  int num_heads_;
  int head_size_;
  float scale_;
  int num_kv_heads_;
  int sliding_window_;

 private:
  inline static StepwiseAttentionState step_wise_attn_state_;
};
TORCH_MODULE(Attention);

}  // namespace layer
}  // namespace xllm
