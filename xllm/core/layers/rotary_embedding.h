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

namespace xllm {
namespace rotary {

float yarn_get_mscale(float scale, float mscale);

torch::Tensor apply_deepseek_yarn_rope_scaling(float factor,
                                               int64_t extrapolation_factor,
                                               int64_t beta_fast,
                                               int64_t beta_slow,
                                               int64_t rotary_dim,
                                               float theta,
                                               int64_t old_context_len);
torch::Tensor compute_inv_freq(int64_t rotary_dim,
                               float rope_theta,
                               const torch::TensorOptions& options);
torch::Tensor compute_cos_sin_cache(int64_t rotary_dim,
                                    int64_t max_position_embeddings,
                                    bool interleaved,
                                    float scaling_factor,
                                    float attn_factor,
                                    float mscale,
                                    float mscale_all_dim,
                                    torch::Tensor inv_freq,
                                    const torch::TensorOptions& options);
torch::Tensor compute_cos_sin_cache(int64_t rotary_dim,
                                    int64_t max_position_embeddings,
                                    bool interleaved,
                                    torch::Tensor inv_freq,
                                    const torch::TensorOptions& options);

// Internal: Cache descriptor structure for cos_sin_cache sharing
struct CosSinCacheDesc {
  int64_t rotary_dim;
  int64_t max_position_embeddings;
  bool interleaved;
  float scaling_factor;
  float attn_factor;
  float mscale;
  float mscale_all_dim;
  size_t inv_freq_hash;  // Hash of inv_freq tensor content
  torch::Device device;
  torch::ScalarType dtype;

  bool operator==(const CosSinCacheDesc& other) const {
    return rotary_dim == other.rotary_dim &&
           max_position_embeddings == other.max_position_embeddings &&
           interleaved == other.interleaved &&
           scaling_factor == other.scaling_factor &&
           attn_factor == other.attn_factor && mscale == other.mscale &&
           mscale_all_dim == other.mscale_all_dim &&
           inv_freq_hash == other.inv_freq_hash && device == other.device &&
           dtype == other.dtype;
  }
};

// Internal: Hash function for CosSinCacheDesc
inline size_t hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

// Internal: Compute hash for inv_freq tensor
size_t compute_inv_freq_hash(const torch::Tensor& inv_freq);

}  // namespace rotary

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
    torch::Tensor inv_freq,
    bool interleaved,
    float& sm_scale,
    const torch::TensorOptions& options);

}  // namespace xllm
