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
#include <variant>

#include "attention_metadata.h"
#include "core/framework/model_context.h"
#include "framework/model/model_args.h"
#include "rotary_embedding_util.h"

namespace xllm {
namespace layer {

class RotaryEmbeddingImpl : public torch::nn::Module {
 public:
  RotaryEmbeddingImpl(int64_t rotary_dim,
                      int64_t max_position_embeddings,
                      int64_t rope_theta,
                      bool interleaved,
                      const torch::TensorOptions& options);
  RotaryEmbeddingImpl(const ModelContext& context);

  void forward(torch::Tensor& q,
               torch::Tensor& k,
               const torch::Tensor& positions,
               const torch::Tensor& cu_query_lens,
               int64_t max_query_len,
               bool is_prompt);

  // Single tensor forward for MLA architecture
  void forward(torch::Tensor& input,
               const torch::Tensor& positions,
               const torch::Tensor& cu_query_lens,
               int64_t max_query_len,
               bool is_prompt);

  torch::Tensor get_cos_sin_cache() { return cos_sin_cache_; }
  torch::Tensor get_sin_cache() { return sin_; }
  torch::Tensor get_cos_cache() { return cos_; }

 protected:
  bool interleaved_;

 private:
  torch::Tensor sin_;
  torch::Tensor cos_;
  torch::Tensor cos_sin_cache_;
};
TORCH_MODULE(RotaryEmbedding);

class MRotaryEmbeddingImpl : public RotaryEmbeddingImpl {
 public:
  MRotaryEmbeddingImpl(int64_t rotary_dim,
                       int64_t max_position_embeddings,
                       int64_t rope_theta,
                       bool interleaved,
                       const std::vector<int64_t>& rope_scaling_mrope_section,
                       const torch::TensorOptions& options);

  void forward(torch::Tensor& q,
               torch::Tensor& k,
               const torch::Tensor& positions,
               const AttentionMetadata& attn_metadata);

 private:
  std::vector<int64_t> mrope_section_;
  torch::Tensor mrope_cu_seq_lens_;
};
TORCH_MODULE(MRotaryEmbedding);

class DeepseekScalingRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  DeepseekScalingRotaryEmbeddingImpl(
      int64_t head_size,
      int64_t rotary_dim,
      int64_t max_position_embeddings,
      int64_t rope_scaling_original_max_position_embeddings,
      int64_t rope_theta,
      bool interleaved,
      float scaling_factor,
      float extrapolation_factor,
      float attn_factor,
      float beta_fast,
      float beta_slow,
      float mscale,
      float mscale_all_dim,
      const torch::TensorOptions& options);

  void forward(torch::Tensor& input,
               const torch::Tensor& positions,
               const torch::Tensor& cu_query_lens,
               int64_t max_query_len,
               bool is_prompt);

  torch::Tensor get_cos_sin_cache() { return cos_sin_cache_; }
  torch::Tensor get_sin_cache() { return sin_; }
  torch::Tensor get_cos_cache() { return cos_; }

 private:
  int64_t head_size_;
  int64_t rotary_dim_;
  bool interleaved_;
  torch::Tensor sin_;
  torch::Tensor cos_;
  torch::Tensor cos_sin_cache_;
};
TORCH_MODULE(DeepseekScalingRotaryEmbedding);

// Unified RoPE variant type for MLA architecture
// Supports both DeepseekScalingRotaryEmbedding (for deepseek_yarn) and
// RotaryEmbedding (for default rope)
// Using shared_ptr wrapper because TORCH_MODULE types have no default
// constructor
using RotaryEmbeddingVariants =
    std::variant<std::shared_ptr<RotaryEmbeddingImpl>,
                 std::shared_ptr<DeepseekScalingRotaryEmbeddingImpl>>;

// Factory function: creates the appropriate RoPE type based on model args
RotaryEmbeddingVariants create_rotary_embedding(
    const ModelArgs& args,
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    bool interleaved,
    const torch::TensorOptions& options);

// Unified forward interface for single tensor (used in MLA architecture)
void apply_rotary_embedding(RotaryEmbeddingVariants& rotary,
                            torch::Tensor& input,
                            const torch::Tensor& positions,
                            const torch::Tensor& cu_query_lens,
                            int64_t max_query_len,
                            bool is_prompt);

// Helper functions to access sin/cos cache from variants
torch::Tensor get_sin_cache(RotaryEmbeddingVariants& rotary);
torch::Tensor get_cos_cache(RotaryEmbeddingVariants& rotary);

}  // namespace layer
}  // namespace xllm
