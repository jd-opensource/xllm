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

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int rotary_dim,
                                         int max_position_embeddings,
                                         int rope_theta,
                                         bool interleaved,
                                         const torch::TensorOptions& options)
    : rotary_dim_(rotary_dim),
      max_position_embeddings_(max_position_embeddings),
      rope_theta_(rope_theta),
      interleaved_(interleaved) {
  auto dev_options = torch::TensorOptions().device(Device::type_torch());

  auto inv_freq_t = torch::arange(/*start=*/0,
                                  /*end=*/rotary_dim_,
                                  /*step=*/2,
                                  torch::TensorOptions().dtype(torch::kFloat));
  inv_freq_t = inv_freq_t.to(dev_options);
  auto inv_freq =
      1.0 /
      torch::pow(rope_theta_, inv_freq_t / static_cast<double>(rotary_dim_));

  auto t = torch::arange(0, max_position_embeddings_, 1, torch::kFloat32);
  t = t.to(dev_options);

  const auto freqs = torch::einsum("i,j->ij", {t, inv_freq});
#if defined(USE_NPU)
  const auto cos_sin =
      torch::cat({freqs.cos(), freqs.sin()}, /*dim=*/-1).contiguous();
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin.to(options));
  auto cos_sin_vec = cos_sin_cache_.chunk(2, /*dim=*/-1);
#else
  // Create cos and sin embeddings.
  torch::Tensor emd;
  if (interleaved) {
    // [a, b, c, d] => [a, a, b, b, c, c, d, d]
    emd = freqs.repeat_interleave(/*repeats=*/2, /*dim=*/-1);
  } else {
    // [a, b, c, d] => [a, b, c, d, a, b, c, d]
    emd = torch::cat({freqs, freqs}, /*dim=*/-1);
  }

  const auto cos_sin = torch::cat({emd.cos(), emd.sin()}, /*dim=*/-1);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin.to(options));

  auto cos_sin_vec = cos_sin_cache_.chunk(2, /*dim=*/-1);
  cos_ = cos_sin_vec[0].view({-1, rotary_dim});
  sin_ = cos_sin_vec[1].view({-1, rotary_dim});
#endif
}

void RotaryEmbeddingImpl::forward(torch::Tensor& q,
                                  torch::Tensor& k,
                                  const torch::Tensor& positions,
                                  const torch::Tensor& cu_query_lens,
                                  int max_query_len,
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
#if defined(USE_NPU)
  rotary_params.positions = positions;
#else
  rotary_params.sin = sin_;
  rotary_params.cos = cos_;
#endif
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

}  // namespace layer
}  // namespace xllm
