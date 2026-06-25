/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "fused_recurrent_gated_delta_rule.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>

#include <cmath>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace kernel {
namespace mlu {

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    const std::optional<torch::Tensor>& beta_opt,
    const std::optional<torch::Tensor>& initial_state_opt,
    bool inplace_final_state,
    const std::optional<torch::Tensor>& cu_seqlens_opt,
    const std::optional<torch::Tensor>& ssm_state_indices_opt,
    const std::optional<torch::Tensor>& num_accepted_tokens_opt,
    bool use_qk_l2norm_in_kernel) {
  torch::Tensor beta = beta_opt.has_value() ? beta_opt.value()
                                            : torch::ones_like(q.select(-1, 0));
  torch::Tensor initial_state = initial_state_opt.value_or(torch::Tensor());
  torch::Tensor cu_seqlens = cu_seqlens_opt.value_or(torch::Tensor());
  torch::Tensor ssm_state_indices =
      ssm_state_indices_opt.value_or(torch::Tensor());
  torch::Tensor num_accepted_tokens =
      num_accepted_tokens_opt.value_or(torch::Tensor());

  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  g = g.contiguous().to(torch::kFloat32);
  beta = beta.contiguous();

  // Set dimensions from input tensors
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t H = k.size(2);
  int64_t K = k.size(3);
  int64_t HV = v.size(2);
  int64_t V = v.size(3);
  int64_t N = cu_seqlens.numel() > 0 ? cu_seqlens.size(0) - 1 : B;

  // Calculate block sizes
  int64_t BK = 1;
  while (BK < K) {
    BK <<= 1;
  }
  int64_t BV = 1;
  while (BV < V) {
    BV <<= 1;
  }
  BV = std::min(BV, static_cast<int64_t>(8));

  // Set strides for indices
  int64_t stride_indices_seq = 1;
  int64_t stride_indices_tok = 1;
  if (ssm_state_indices.numel() > 0) {
    if (ssm_state_indices.ndimension() == 1) {
      stride_indices_seq = ssm_state_indices.stride(0);
      stride_indices_tok = 1;
    } else {
      stride_indices_seq = ssm_state_indices.stride(0);
      stride_indices_tok = ssm_state_indices.stride(1);
    }
  }

  // Set configuration flags
  bool is_kda = false;
  bool is_beta_headwise = (beta.ndimension() == v.ndimension());

  // Create output tensor
  torch::Tensor o = torch::empty_like(v);

  // Set final state
  torch::Tensor ht;
  torch::Tensor h0;
  if (inplace_final_state && initial_state.numel() > 0) {
    ht = initial_state;
    h0 = initial_state;
  } else {
    auto dtype = initial_state.numel() > 0 ? initial_state.dtype() : v.dtype();
    ht = torch::empty({T, HV, V, K},
                      torch::TensorOptions().dtype(dtype).device(v.device()));
    if (initial_state.numel() > 0) {
      h0 = initial_state;
    }
  }
  ht = ht.to(torch::kFloat32);
  h0 = h0.to(torch::kFloat32);

  // Set strides
  int64_t stride_init_state_token = h0.numel() > 0 ? h0.stride(0) : 0;
  int64_t stride_final_state_token = ht.stride(0);
  float scale = 1.0f / std::sqrt(static_cast<float>(k.size(-1)));
  auto queue = torch_mlu::getCurMLUStream();

  // Grid calculation: (NV, N * HV)
  int64_t NV = (V + BV - 1) / BV;
  int32_t num_programs_x = static_cast<int32_t>(NV);
  int32_t num_programs_y = static_cast<int32_t>(N * HV);
  cnrtDim3_t dim_block = {static_cast<uint32_t>(num_programs_x),
                          static_cast<uint32_t>(num_programs_y),
                          1};

  auto beta_ptr = beta_opt.has_value() ? beta_opt.value().data_ptr() : nullptr;
  auto cu_seqlens_ptr =
      cu_seqlens_opt.has_value() ? cu_seqlens_opt.value().data_ptr() : nullptr;
  auto ssm_state_indices_ptr = ssm_state_indices_opt.has_value()
                                   ? ssm_state_indices_opt.value().data_ptr()
                                   : nullptr;
  auto num_accepted_tokens_ptr =
      num_accepted_tokens_opt.has_value()
          ? num_accepted_tokens_opt.value().data_ptr()
          : nullptr;

  constexpr int32_t algo_id = 0;
  tmo_fused_recurrent_gated_delta_rule_fwd_kernel(
      queue,
      &dim_block,
      q.data_ptr(),
      k.data_ptr(),
      v.data_ptr(),
      g.data_ptr(),
      beta_ptr,
      o.data_ptr(),
      h0.data_ptr(),
      ht.data_ptr(),
      cu_seqlens_ptr,
      ssm_state_indices_ptr,
      num_accepted_tokens_ptr,
      scale,
      static_cast<int32_t>(N),
      static_cast<int32_t>(T),
      static_cast<int32_t>(B),
      static_cast<int32_t>(H),
      static_cast<int32_t>(HV),
      static_cast<int32_t>(K),
      static_cast<int32_t>(V),
      static_cast<int32_t>(stride_init_state_token),
      static_cast<int32_t>(stride_final_state_token),
      static_cast<int32_t>(stride_indices_seq),
      static_cast<int32_t>(stride_indices_tok),
      algo_id);

  return std::make_pair(o, ht);
}

}  // namespace mlu
}  // namespace kernel
}  // namespace xllm
