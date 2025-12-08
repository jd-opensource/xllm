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

#include "rotary_embedding_util.h"

#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <cmath>
#include <memory>

#include "core/common/global_flags.h"
#include "core/util/slice.h"

namespace xllm {

namespace {

// Hash function for CosSinCacheDesc
struct CosSinCacheDescHash {
  size_t operator()(const layer::rotary::CosSinCacheDesc& key) const {
    size_t hash = 0;
    hash =
        layer::rotary::hash_combine(hash, std::hash<int64_t>{}(key.rotary_dim));
    hash = layer::rotary::hash_combine(
        hash, std::hash<int64_t>{}(key.max_position_embeddings));
    hash =
        layer::rotary::hash_combine(hash, std::hash<bool>{}(key.interleaved));
    hash = layer::rotary::hash_combine(hash,
                                       std::hash<float>{}(key.scaling_factor));
    hash =
        layer::rotary::hash_combine(hash, std::hash<float>{}(key.attn_factor));
    hash = layer::rotary::hash_combine(hash, std::hash<float>{}(key.mscale));
    hash = layer::rotary::hash_combine(hash,
                                       std::hash<float>{}(key.mscale_all_dim));
    hash = layer::rotary::hash_combine(hash, key.inv_freq_hash);
    hash = layer::rotary::hash_combine(hash,
                                       std::hash<torch::Device>{}(key.device));
    hash = layer::rotary::hash_combine(
        hash, std::hash<torch::ScalarType>{}(key.dtype));
    return hash;
  }
};

// Cache manager for cos_sin_cache sharing
class CosSinCacheManager {
 public:
  static CosSinCacheManager& get_instance() {
    static CosSinCacheManager instance;
    return instance;
  }

  torch::Tensor get_or_create_cache(const layer::rotary::CosSinCacheDesc& key,
                                    std::function<torch::Tensor()> factory) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;  // Return shared tensor
    }
    // Create new and cache it
    auto tensor = factory();
    cache_[key] = tensor;
    return tensor;
  }

 private:
  std::unordered_map<layer::rotary::CosSinCacheDesc,
                     torch::Tensor,
                     CosSinCacheDescHash>
      cache_;
};

using torch::indexing::None;
using ISlice = torch::indexing::Slice;

// Inverse dim formula to find dim based on number of rotations
inline double yarn_find_correction_dim(int num_rotations,
                                       int dim,
                                       float theta,
                                       int max_position_embeddings) {
  return (dim *
          std::log(max_position_embeddings / (num_rotations * 2 * M_PI))) /
         (2 * std::log(theta));
}

// Find dim range bounds based on rotations
inline std::tuple<int, int> yarn_find_correction_range(
    int low_rot,
    int high_rot,
    int dim,
    float theta,
    int max_position_embeddings) {
  int low = std::floor(
      yarn_find_correction_dim(low_rot, dim, theta, max_position_embeddings));
  int high = std::ceil(
      yarn_find_correction_dim(high_rot, dim, theta, max_position_embeddings));
  return std::make_tuple(std::max(low, 0), std::min(high, dim - 1));
}

// Create cos_sin tensor for rotary embedding cache
inline torch::Tensor create_cos_sin_tensor(
    int64_t max_position_embeddings,
    float scaling_factor,
    float attn_factor,
    float mscale,
    float mscale_all_dim,
    bool interleaved,
    const torch::Tensor& inv_freq,
    const torch::TensorOptions& options) {
  float mscale_ = static_cast<float>(
      layer::rotary::yarn_get_mscale(scaling_factor, mscale) /
      layer::rotary::yarn_get_mscale(scaling_factor, mscale_all_dim) *
      attn_factor);
  // [max_position_embeddings]
  auto t = torch::arange(max_position_embeddings * scaling_factor);
  // [max_position_embeddings, rotary_dim/2]
  const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
  // Create cos and sin embeddings.
  torch::Tensor emd;
  if (interleaved) {
    // [a, b, c, d] => [a, a, b, b, c, c, d, d]
    emd = freqs.repeat_interleave(/*repeats=*/2, /*dim=*/-1);
  } else {
    // [a, b, c, d] => [a, b, c, d, a, b, c, d]
    emd = torch::cat({freqs, freqs}, /*dim=*/-1);
  }
  const auto cos_sin =
      torch::cat({emd.cos() * mscale_, emd.sin() * mscale_}, /*dim=*/-1)
          .to(options);
  return cos_sin;
}

inline torch::Tensor yarn_linear_ramp_mask(float low, float high, int64_t dim) {
  if (low == high) {
    high += 0.001;  // Prevent singularity
  }
  torch::Tensor linear_func = torch::arange(dim) - low;
  linear_func = linear_func / (high - low);
  torch::Tensor ramp_func = torch::clamp(linear_func, 0, 1);
  return ramp_func;
}

torch::Tensor compute_rotary_embedding(int64_t dim,
                                       int64_t seq_len,
                                       double rope_theta,
                                       const torch::TensorOptions& options,
                                       bool use_cat) {
  auto options_new =
      torch::device(options.device()).dtype(at::ScalarType::Double);
  auto inv_freq =
      1.0 / torch::pow(rope_theta, torch::arange(0, dim, 2, options_new) / dim)
                .to(at::ScalarType::Float);
  auto seq_idx = torch::arange(seq_len, options_new);

  auto freqs = torch::ger(seq_idx, inv_freq).to(torch::kFloat32);
  torch::Tensor emb;
  if (use_cat) {
    emb = torch::cat({freqs, freqs}, -1);
  } else {
    emb = torch::stack({freqs, freqs}, -1);
    emb = emb.reshape({seq_len, dim});
  }
  auto rope_cos = torch::cos(emb);
  auto rope_sin = torch::sin(emb);

  auto dtype = options.dtype();
  if (dtype == torch::kFloat16 || dtype == torch::kBFloat16 ||
      dtype == torch::kInt8) {
    if (dtype == torch::kBFloat16) {
      rope_cos = rope_cos.to(torch::kBFloat16);
      rope_sin = rope_sin.to(torch::kBFloat16);
    } else {
      rope_cos = rope_cos.to(torch::kFloat16);
      rope_sin = rope_sin.to(torch::kFloat16);
    }
  }
  std::vector<torch::Tensor> cos_sin{rope_cos, rope_sin};
  return torch::cat(cos_sin, -1);
}

}  // namespace

namespace layer {
namespace rotary {

// Compute hash for inv_freq tensor content
size_t compute_inv_freq_hash(const torch::Tensor& inv_freq) {
  // Ensure tensor is contiguous and on CPU for hashing
  auto cpu_tensor = inv_freq.contiguous().cpu();
  auto numel = cpu_tensor.numel();
  if (numel == 0) {
    return 0;
  }

  // Compute hash based on tensor data
  size_t hash = 0;
  auto data_ptr = cpu_tensor.data_ptr<float>();
  for (int64_t i = 0; i < numel; ++i) {
    // Hash each float value
    hash = hash_combine(hash, std::hash<float>{}(data_ptr[i]));
  }
  return hash;
}

float yarn_get_mscale(float scale, float mscale) {
  if (scale <= 0.0) {
    return 1.0;
  }
  return 0.1 * mscale * std::log(scale) + 1.0;
}

torch::Tensor apply_deepseek_yarn_rope_scaling(float factor,
                                               int64_t extrapolation_factor,
                                               int64_t beta_fast,
                                               int64_t beta_slow,
                                               int64_t rotary_dim,
                                               float theta,
                                               int64_t old_context_len) {
  CHECK(rotary_dim % 2 == 0) << "rotary_dim must be even";
  const auto slice = torch::arange(0, rotary_dim, 2, torch::kFloat32);
  torch::Tensor pos_freqs = torch::pow(theta, slice / rotary_dim);
  torch::Tensor inv_freq_extrapolation = 1.0 / pos_freqs;
  torch::Tensor inv_freq_interpolation = 1.0 / (factor * pos_freqs);
  int low, high;
  std::tie(low, high) = yarn_find_correction_range(
      beta_fast, beta_slow, rotary_dim, theta, old_context_len);
  // Get n-d rotational scaling corrected for extrapolation
  torch::Tensor inv_freq_mask =
      1 - yarn_linear_ramp_mask(low, high, rotary_dim / 2);
  inv_freq_mask = inv_freq_mask * extrapolation_factor;
  torch::Tensor inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) +
                           inv_freq_extrapolation * inv_freq_mask;
  return inv_freq;
}

torch::Tensor compute_inv_freq(int64_t rotary_dim,
                               float rope_theta,
                               const torch::TensorOptions& options) {
  const auto slice = torch::arange(0, rotary_dim, 2, torch::kFloat32);
  torch::Tensor inv_freq =
      1.0 / torch::pow(rope_theta, slice / static_cast<double>(rotary_dim));
  return inv_freq;
}

torch::Tensor compute_cos_sin_cache(int64_t rotary_dim,
                                    int64_t max_position_embeddings,
                                    bool interleaved,
                                    float scaling_factor,
                                    float attn_factor,
                                    float mscale,
                                    float mscale_all_dim,
                                    torch::Tensor inv_freq,
                                    const torch::TensorOptions& options) {
  // Create cache descriptor using aggregate initialization
  CosSinCacheDesc desc{rotary_dim,
                       max_position_embeddings,
                       interleaved,
                       scaling_factor,
                       attn_factor,
                       mscale,
                       mscale_all_dim,
                       compute_inv_freq_hash(inv_freq),
                       options.device(),
                       options.dtype().toScalarType()};

  // Get or create cache using cache manager
  auto& cache_manager = CosSinCacheManager::get_instance();
  return cache_manager.get_or_create_cache(desc, [&]() {
    return create_cos_sin_tensor(max_position_embeddings,
                                 scaling_factor,
                                 attn_factor,
                                 mscale,
                                 mscale_all_dim,
                                 interleaved,
                                 inv_freq,
                                 options);
  });
}

torch::Tensor compute_cos_sin_cache(int64_t rotary_dim,
                                    int64_t max_position_embeddings,
                                    bool interleaved,
                                    torch::Tensor inv_freq,
                                    const torch::TensorOptions& options) {
  return compute_cos_sin_cache(rotary_dim,
                               max_position_embeddings,
                               interleaved,
                               1.0f,
                               1.0f,
                               1.0f,
                               1.0f,
                               inv_freq,
                               options);
}

torch::Tensor get_concat_rotary_embedding(int64_t dim,
                                          int64_t seq_len,
                                          double rope_theta,
                                          const torch::TensorOptions& options) {
  return compute_rotary_embedding(dim, seq_len, rope_theta, options, true);
}

torch::Tensor get_chatglm_rotary_embedding(
    int64_t dim,
    int64_t seq_len,
    double rope_theta,
    const torch::TensorOptions& options) {
  return compute_rotary_embedding(dim, seq_len, rope_theta, options, false);
}

}  // namespace rotary
}  // namespace layer
}  // namespace xllm
