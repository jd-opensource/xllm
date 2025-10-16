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

#include <optional>
#include <string>
#include <tuple>

#include "attention.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"
#include "linear.h"
#include "rotary_embedding.h"

namespace xllm {
namespace layer {

class IndexerImpl : public torch::nn::Module {
 public:
  IndexerImpl() = default;

  IndexerImpl(int dim,
              int index_n_heads,
              int index_head_dim,
              int qk_rope_head_dim,
              int index_topk,
              int q_lora_rank,
              int max_position_embeddings,
              double rope_theta,
              bool rope_interleaved,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& x,
      const torch::Tensor& qr,
      const torch::Tensor& positions,
      torch::Tensor& k_cache,
      const AttentionMetadata& attn_metadata,
      bool is_prefill,
      const std::optional<torch::Tensor>& mask = std::nullopt);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  // whether the weight is loaded
  void verify_loaded_weights(const std::string& prefix = "") const;

 private:
  int dim_;
  int n_heads_;
  int head_dim_;
  int rope_head_dim_;
  int index_topk_;
  int q_lora_rank_;
  float softmax_scale_;
  float hadamard_transform_scale_;

  // Linear layers using ReplicatedLinear
  ReplicatedLinear wq_b_{nullptr};
  ReplicatedLinear wk_{nullptr};
  ReplicatedLinear weights_proj_{nullptr};
  torch::nn::LayerNorm k_norm_{nullptr};

  // Rotary embedding
  RotaryEmbedding rotary_emb_{nullptr};

  // Hadamard matrix
  torch::Tensor hadamard_matrix_;

  // Helper function for rotation activation
  torch::Tensor rotate_activation(const torch::Tensor& input,
                                  const torch::Tensor& hadamard_matrix);
};

TORCH_MODULE(Indexer);

}  // namespace layer
}  // namespace xllm
