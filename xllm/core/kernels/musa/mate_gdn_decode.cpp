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

#include "mate_gdn_ops.h"

#include <sstream>

#include "core/kernels/param.h"
#include "cuda_ops_api.h"
#include "utils.h"

namespace xllm::kernel::cuda {
namespace {

std::string mate_gdn_dtype_suffix(torch::ScalarType dtype) {
  if (dtype == torch::kBFloat16) {
    return "bf16";
  }
  if (dtype == torch::kFloat16) {
    return "f16";
  }
  TORCH_CHECK(false, "mate GDN decode expects bfloat16 or float16 q/k/v");
}

}  // namespace

std::string get_mate_gdn_decode_uri(int64_t num_q_heads,
                                    int64_t num_v_heads,
                                    torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_decode_hq" << num_q_heads << "_hv" << num_v_heads << "_"
      << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

torch::Tensor mate_gated_delta_rule_decode(
    MateGatedDeltaRuleDecodeParams& params) {
  auto mixed_qkv = params.mixed_qkv.contiguous();
  TORCH_CHECK(mixed_qkv.dim() == 2, "mate GDN decode expects mixed_qkv [B, D]");
  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t num_k_heads = params.num_k_heads;
  const int64_t num_v_heads = params.num_v_heads;
  const int64_t head_k_dim = params.head_k_dim;
  const int64_t head_v_dim = params.head_v_dim;
  const int64_t qk_cols = num_k_heads * head_k_dim;
  const int64_t v_cols = num_v_heads * head_v_dim;
  TORCH_CHECK(mixed_qkv.size(1) == 2 * qk_cols + v_cols,
              "mate GDN decode mixed_qkv dim mismatch");

  auto query = mixed_qkv.slice(/*dim=*/1, /*start=*/0, /*end=*/qk_cols)
                   .reshape({batch_size, num_k_heads, head_k_dim})
                   .contiguous();
  auto key =
      mixed_qkv.slice(/*dim=*/1, /*start=*/qk_cols, /*end=*/2 * qk_cols)
          .reshape({batch_size, num_k_heads, head_k_dim})
          .contiguous();
  // NOTE: end=-1 would exclude the last column (Python-style negative index)
  // and shrink the slice to (v_cols - 1) elements, breaking the reshape to
  // [B, num_v_heads, head_v_dim]. Use the explicit end to capture the full
  // value range. xllm_0526 reference behaves the same way.
  auto value =
      mixed_qkv
          .slice(/*dim=*/1, /*start=*/2 * qk_cols, /*end=*/2 * qk_cols + v_cols)
          .reshape({batch_size, num_v_heads, head_v_dim})
          .contiguous();

  auto a = params.a.contiguous();
  auto b = params.b.contiguous();
  if (a.dim() == 1) {
    a = a.unsqueeze(0);
  }
  if (b.dim() == 1) {
    b = b.unsqueeze(0);
  }
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
              "mate GDN decode expects a/b shaped [B, Hv]");

  auto state_f32 = params.state.to(torch::kFloat32).contiguous();
  auto state_indices = params.state_indices.to(torch::kInt32).contiguous();
  auto output =
      params.decode_output.has_value() && params.decode_output.value().defined()
          ? params.decode_output.value()
          : torch::empty({batch_size, num_v_heads, head_v_dim},
                         value.options());

  const std::string uri = get_mate_gdn_decode_uri(
      num_k_heads, num_v_heads, query.scalar_type());
  bind_tvmffi_stream_to_current_torch_stream(query.device());
  auto run = get_function(uri, "run");

  run(to_ffi_tensor(query),
      to_ffi_tensor(key),
      to_ffi_tensor(value),
      to_ffi_tensor(params.A_log.contiguous()),
      to_ffi_tensor(a),
      to_ffi_tensor(params.dt_bias.contiguous()),
      to_ffi_tensor(b),
      to_ffi_tensor(state_indices),
      to_ffi_tensor(state_f32),
      to_ffi_tensor(output));

  auto updated_state =
      state_f32.index_select(/*dim=*/0, state_indices.to(torch::kLong));
  params.state.index_copy_(
      /*dim=*/0, state_indices.to(torch::kLong),
      updated_state.to(params.state.scalar_type()));

  return output;
}

}  // namespace xllm::kernel::cuda
