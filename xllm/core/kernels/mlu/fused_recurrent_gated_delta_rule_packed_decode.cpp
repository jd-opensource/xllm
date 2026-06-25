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

#include "fused_recurrent_gated_delta_rule_packed_decode.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <glog/logging.h>

#include <cmath>

#include "kernels/mlu/mlu_ops_api.h"

namespace xllm {
namespace kernel {
namespace mlu {

std::pair<torch::Tensor, torch::Tensor>
fused_recurrent_gated_delta_rule_packed_decode(
    torch::Tensor& mixed_qkv,
    torch::Tensor& a,
    torch::Tensor& b,
    const torch::Tensor& A_log,
    const torch::Tensor& dt_bias,
    double scale,
    torch::Tensor& ssm_cache,
    const torch::Tensor& ssm_state_indices,
    bool use_qk_l2norm_in_kernel) {
  mixed_qkv = mixed_qkv.contiguous();
  a = a.contiguous();
  b = b.contiguous();

  int64_t B = mixed_qkv.size(0);
  int64_t qkv_dim = mixed_qkv.size(1);
  int64_t HV = ssm_cache.size(1);
  int64_t V = ssm_cache.size(2);
  int64_t K = ssm_cache.size(3);
  int64_t qk_dim = qkv_dim - HV * V;
  int64_t H = qk_dim / (2 * K);

  int64_t BK = 1;
  while (BK < K) {
    BK <<= 1;
  }
  BK = std::min(BK, static_cast<int64_t>(128));
  int64_t BV = 128;

  // Create output tensor: [B, 1, HV, V] in same dtype as mixed_qkv
  torch::Tensor out =
      torch::empty({B, 1, HV, V}, mixed_qkv.options().dtype(mixed_qkv.dtype()));

  // Strides
  int64_t stride_mixed_qkv_tok = mixed_qkv.stride(0);
  int64_t stride_a_tok = a.stride(0);
  int64_t stride_b_tok = b.stride(0);
  int64_t stride_init_state_token = ssm_cache.stride(0);
  int64_t stride_final_state_token = ssm_cache.stride(0);
  int64_t stride_indices_seq = ssm_state_indices.stride(0);

  // Grid: (NV, B * HV)
  int64_t NV = (V + BV - 1) / BV;
  int32_t num_programs_x = static_cast<int32_t>(NV);
  int32_t num_programs_y = static_cast<int32_t>(B * HV);
  cnrtDim3_t dim_block = {static_cast<uint32_t>(num_programs_x),
                          static_cast<uint32_t>(num_programs_y),
                          1};

  auto queue = torch_mlu::getCurMLUStream();

  constexpr int32_t algo_id = 0;

  tmo_fused_recurrent_gated_delta_rule_packed_decode_kernel(
      queue,
      &dim_block,
      mixed_qkv.data_ptr(),
      a.data_ptr(),
      b.data_ptr(),
      A_log.data_ptr(),
      dt_bias.data_ptr(),
      out.data_ptr(),
      ssm_cache.data_ptr(),  // h0 (float32, [num_slots, HV, V, K])
      ssm_cache.data_ptr(),  // ht (same tensor, in-place update)
      ssm_state_indices.data_ptr(),
      static_cast<float>(scale),
      static_cast<int32_t>(stride_mixed_qkv_tok),
      static_cast<int32_t>(stride_a_tok),
      static_cast<int32_t>(stride_b_tok),
      static_cast<int32_t>(stride_init_state_token),
      static_cast<int32_t>(stride_final_state_token),
      static_cast<int32_t>(stride_indices_seq),
      static_cast<int32_t>(H),
      static_cast<int32_t>(HV),
      static_cast<int32_t>(K),
      static_cast<int32_t>(V),
      algo_id);

  return std::make_pair(out, ssm_cache);
}

}  // namespace mlu
}  // namespace kernel
}  // namespace xllm
