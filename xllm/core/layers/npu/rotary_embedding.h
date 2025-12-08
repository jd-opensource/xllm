/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include <functional>
#include <tuple>
#include <unordered_map>

#include "framework/model/model_args.h"
#include "layers/common/rotary_embedding_util.h"

namespace xllm {

class RotaryEmbedding : public torch::nn::Module {
 public:
  ~RotaryEmbedding() override = default;

  // returns a tuple of query and key embeddings with the same shape as the
  // input query and key.
  virtual std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const = 0;

  virtual torch::Tensor get_cos_sin_cache() = 0;
};

class RotaryEmbeddingGeneric : public RotaryEmbedding {
 public:
  RotaryEmbeddingGeneric(int64_t rotary_dim,
                         int64_t max_position_embeddings,
                         torch::Tensor inv_freq,
                         bool interleaved,
                         const torch::TensorOptions& options);

  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const override;

  torch::Tensor get_cos_sin_cache() override { return cos_sin_cache_; }

 private:
  torch::Tensor cos_sin_cache_;

  int64_t rotary_dim_ = 0;

  bool interleaved_ = false;
};

class RotaryEmbeddingDeepseekYarn : public RotaryEmbedding {
 public:
  RotaryEmbeddingDeepseekYarn(float scaling_factor,
                              int64_t rotary_dim,
                              int64_t max_position_embeddings,
                              bool interleaved,
                              float attn_factor,
                              float mscale,
                              float mscale_all_dim,
                              torch::Tensor inv_freq,
                              const torch::TensorOptions& options);
  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const override;

  torch::Tensor get_cos_sin_cache() override { return cos_sin_cache_; }

 private:
  torch::Tensor cos_sin_cache_;
  int64_t rotary_dim_ = 0;
  bool interleaved_ = false;
};

// Rotary Embedding with Multimodal Sections.
class MRotaryEmbedding : public RotaryEmbedding {
 public:
  MRotaryEmbedding(int64_t rotary_dim,
                   int64_t max_position_embeddings,
                   torch::Tensor inv_freq,
                   bool interleaved,
                   const std::vector<int64_t>& mrope_section,
                   const torch::TensorOptions& options);

  // inplace rotary positional embedding
  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& query,     // [num_tokens, n_heads, head_dim]
      const torch::Tensor& key,       // [num_tokens, n_kv_heads, head_dim]
      const torch::Tensor& positions  // [num_tokens]
  ) const override;

  torch::Tensor get_cos_sin_cache() override { return cos_sin_cache_; }

 private:
  torch::Tensor cos_sin_cache_;
  int64_t rotary_dim_ = 0;
  bool interleaved_ = false;
  std::vector<int64_t> mrope_section_;
};

std::shared_ptr<RotaryEmbedding> create_rotary_embedding(
    const ModelArgs& model_args,
    int64_t rotary_dim,
    bool interleaved,
    const torch::TensorOptions& options);

}  // namespace xllm
