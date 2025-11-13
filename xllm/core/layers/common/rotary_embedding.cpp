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

#include "rotary_embedding.h"

#include "kernels/ops_api.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int64_t rotary_dim,
                                         int64_t max_position_embeddings,
                                         int64_t rope_theta,
                                         bool interleaved,
                                         const torch::TensorOptions& options)
    : interleaved_(interleaved) {
  auto inv_freq =
      xllm::rotary::compute_inv_freq(rotary_dim, rope_theta, options);
  const auto cos_sin = xllm::rotary::compute_cos_sin_cache(
      rotary_dim, max_position_embeddings, interleaved, inv_freq, options);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin);

  auto cos_sin_vec = cos_sin_cache_.chunk(2, /*dim=*/-1);
  cos_ = cos_sin_vec[0].view({-1, rotary_dim});
  sin_ = cos_sin_vec[1].view({-1, rotary_dim});
}

void RotaryEmbeddingImpl::forward(torch::Tensor& q,
                                  torch::Tensor& k,
                                  const torch::Tensor& positions,
                                  const torch::Tensor& cu_query_lens,
                                  int64_t max_query_len,
                                  bool is_prompt) {
  bool discrete;
  std::optional<torch::Tensor> position_ids;
  if (is_prompt) {
    discrete = false;
    if (Device::type_str() == "cuda") {
      position_ids = positions;
    }
  } else {
    discrete = true;
    position_ids = positions;
  }

  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q;
  rotary_params.k = k;
  rotary_params.sin = sin_;
  rotary_params.cos = cos_;
  rotary_params.cos_sin = cos_sin_cache_;
  rotary_params.position_ids = position_ids;
  rotary_params.cu_query_lens = cu_query_lens;
  rotary_params.interleaved = interleaved_;
  rotary_params.discrete = discrete;
  rotary_params.max_query_len = max_query_len;
  xllm::kernel::apply_rotary(rotary_params);

  q = rotary_params.q;
  k = rotary_params.k;
}

MRotaryEmbeddingImpl::MRotaryEmbeddingImpl(
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    int64_t rope_theta,
    bool interleaved,
    const std::vector<int64_t>& rope_scaling_mrope_section,
    const torch::TensorOptions& options)
    : RotaryEmbeddingImpl(rotary_dim,
                          max_position_embeddings,
                          rope_theta,
                          interleaved,
                          options),
      interleaved_(interleaved),
      mrope_section_(rope_scaling_mrope_section) {
  mrope_cu_seq_lens_ = torch::zeros(2, torch::kInt32).to(options.device());
}

void MRotaryEmbeddingImpl::precompute_sin_cos_cache(
    const torch::Tensor& positions,
    AttentionMetadata& attn_metadata) {
  int32_t total_len = positions.size(-1);
  mrope_cu_seq_lens_[1] = total_len;
  if (attn_metadata.mrope_cos.defined() && attn_metadata.mrope_sin.defined()) {
    return;
  }

  auto ndim = positions.dim();
  CHECK(ndim == 1 || ndim == 2) << "positions must be 1D or 2D tensor";
  auto cos_sin = get_cos_sin_cache().index({positions});
  auto chunks = cos_sin.chunk(2, -1);
  auto cos = chunks[0];
  auto sin = chunks[1];

  if (positions.dim() == 2) {
    TORCH_CHECK(!mrope_section_.empty(), "mrope_section must not be empty");
    std::vector<int64_t> repeated_sections(mrope_section_);
    repeated_sections.insert(
        repeated_sections.end(), mrope_section_.begin(), mrope_section_.end());

    const auto apply_multi_rope =
        [repeated_sections,
         num_sections = mrope_section_.size()](torch::Tensor tensor) {
          auto splits = tensor.split(repeated_sections, -1);
          std::vector<torch::Tensor> processed;

          for (size_t i = 0; i < splits.size(); ++i) {
            processed.push_back(splits[i][i % num_sections]);
          }
          return torch::cat(processed, -1).contiguous();
        };

    cos = apply_multi_rope(cos);
    sin = apply_multi_rope(sin);
  }

  attn_metadata.mrope_cos = std::move(cos);
  attn_metadata.mrope_sin = std::move(sin);
}

void MRotaryEmbeddingImpl::forward(torch::Tensor& q,
                                   torch::Tensor& k,
                                   const torch::Tensor& positions,
                                   AttentionMetadata& attn_metadata) {
  bool only_prefill =
      (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);
  if (!only_prefill || mrope_section_.empty()) {
    torch::Tensor position_ids = positions;
    if (positions.dim() == 2) {
      position_ids = positions[0];
    }
    return RotaryEmbeddingImpl::forward(q,
                                        k,
                                        position_ids,
                                        attn_metadata.query_start_loc,
                                        attn_metadata.max_query_len,
                                        attn_metadata.is_prefill);
  }

  precompute_sin_cos_cache(positions, attn_metadata);
  int64_t num_tokens = positions.size(-1);
  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q;
  rotary_params.k = k;
  rotary_params.sin = attn_metadata.mrope_sin;
  rotary_params.cos = attn_metadata.mrope_cos;
  rotary_params.cos_sin = get_cos_sin_cache();
  rotary_params.position_ids = std::nullopt;
  rotary_params.cu_query_lens = mrope_cu_seq_lens_;
  rotary_params.interleaved = interleaved_;
  rotary_params.discrete = false;
  rotary_params.max_query_len = num_tokens;
  xllm::kernel::apply_rotary(rotary_params);

  q = rotary_params.q;
  k = rotary_params.k;
}

DeepseekScalingRotaryEmbeddingImpl::DeepseekScalingRotaryEmbeddingImpl(
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
    const torch::TensorOptions& options)
    : head_size_(head_size),
      rotary_dim_(rotary_dim),
      interleaved_(interleaved) {
  auto inv_freq = xllm::rotary::apply_deepseek_yarn_rope_scaling(
      scaling_factor,
      extrapolation_factor,
      beta_fast,
      beta_slow,
      rotary_dim,
      rope_theta,
      rope_scaling_original_max_position_embeddings);
  const auto cos_sin =
      xllm::rotary::compute_cos_sin_cache(rotary_dim,
                                          max_position_embeddings,
                                          interleaved,
                                          scaling_factor,
                                          attn_factor,
                                          mscale,
                                          mscale_all_dim,
                                          inv_freq,
                                          options);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin);

  auto cos_sin_vec = cos_sin_cache_.chunk(2, /*dim=*/-1);
  cos_ = cos_sin_vec[0].view({-1, rotary_dim});
  sin_ = cos_sin_vec[1].view({-1, rotary_dim});
}

void DeepseekScalingRotaryEmbeddingImpl::forward(
    torch::Tensor& q,
    torch::Tensor& k,
    const torch::Tensor& positions,
    const torch::Tensor& cu_query_lens,
    int64_t max_query_len,
    bool is_prompt) {
  bool discrete;
  std::optional<torch::Tensor> position_ids;
  if (is_prompt) {
    discrete = false;
    position_ids = std::nullopt;
  } else {
    discrete = true;
    position_ids = positions;
    max_query_len = 1;
  }

  auto q_rot = q.slice(-1, 0, rotary_dim_);
  auto k_rot = k.slice(-1, 0, rotary_dim_);
  torch::Tensor q_pass, k_pass;
  if (rotary_dim_ < head_size_) {
    q_pass = q.slice(-1, rotary_dim_, head_size_);
    k_pass = k.slice(-1, rotary_dim_, head_size_);
  }

  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q_rot;
  rotary_params.sin = sin_;
  rotary_params.cos = cos_;
  rotary_params.cos_sin = cos_sin_cache_;
  rotary_params.position_ids = position_ids;
  rotary_params.cu_query_lens = cu_query_lens;
  rotary_params.interleaved = interleaved_;
  rotary_params.discrete = discrete;
  rotary_params.max_query_len = max_query_len;
  xllm::kernel::apply_rotary(rotary_params);
  q_rot = rotary_params.q;

  rotary_params.q = k_rot;
  xllm::kernel::apply_rotary(rotary_params);
  k_rot = rotary_params.q;
  if (rotary_dim_ < head_size_) {
    q = torch::cat({q_rot, q_pass}, -1);
    k = torch::cat({k_rot, k_pass}, -1);
  } else {
    q = q_rot;
    k = k_rot;
  }
}

}  // namespace layer
}  // namespace xllm
