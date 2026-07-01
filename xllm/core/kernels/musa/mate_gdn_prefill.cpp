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

constexpr int64_t kGdnChunkSize = 64;

int64_t chunk_pad_size(int64_t seq_len, int64_t chunk_size) {
  return (chunk_size - seq_len % chunk_size) % chunk_size;
}

torch::Tensor pad_time_dim_4d(const torch::Tensor& tensor, int64_t pad_size) {
  if (pad_size == 0) {
    return tensor;
  }
  return torch::nn::functional::pad(
      tensor,
      torch::nn::functional::PadFuncOptions({0, 0, 0, 0, 0, pad_size}));
}

torch::Tensor pad_time_dim_3d(const torch::Tensor& tensor,
                              int64_t pad_size,
                              double pad_value) {
  if (pad_size == 0) {
    return tensor;
  }
  return torch::nn::functional::pad(
      tensor,
      torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size})
          .mode(torch::kConstant)
          .value(pad_value));
}

std::string mate_gdn_dtype_suffix(torch::ScalarType dtype) {
  if (dtype == torch::kBFloat16) {
    return "bf16";
  }
  if (dtype == torch::kFloat16) {
    return "f16";
  }
  TORCH_CHECK(false, "mate GDN prefill expects bfloat16 or float16 q/k/v");
}

void l2norm_last_dim(torch::Tensor& tensor) {
  const auto orig_dtype = tensor.scalar_type();
  tensor = torch::nn::functional::normalize(
      tensor.to(torch::kFloat32),
      torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
  tensor = tensor.to(orig_dtype);
}

// xLLM fused gating emits log(alpha). MATE expects per-chunk cumulative log
// alpha along the sequence axis (see mate.gdn_prefill.chunk_local_cumsum).
torch::Tensor chunk_local_cumsum_log_alpha(const torch::Tensor& g_log,
                                           int64_t chunk_size) {
  auto alpha = g_log.to(torch::kFloat32).exp().contiguous();
  const int64_t batch_size = alpha.size(0);
  const int64_t pad_size = chunk_pad_size(alpha.size(1), chunk_size);
  if (pad_size > 0) {
    alpha = pad_time_dim_3d(alpha, pad_size, 1.0);
  }
  const int64_t padded_len = alpha.size(1);
  auto log_alpha = alpha.clamp_min(1e-20f).log();
  log_alpha =
      log_alpha.reshape({batch_size, padded_len / chunk_size, chunk_size, -1});
  log_alpha = log_alpha.cumsum(/*dim=*/2);
  return log_alpha.reshape({batch_size, padded_len, alpha.size(2)}).contiguous();
}

}  // namespace

std::string get_mate_gdn_prefill_uri(int64_t num_q_heads,
                                     int64_t num_v_heads,
                                     torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_hq" << num_q_heads << "_hv" << num_v_heads << "_"
      << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

std::pair<torch::Tensor, torch::Tensor> mate_gated_delta_rule_prefill(
    MateGatedDeltaRulePrefillParams& params) {
  auto query = params.q.contiguous();
  auto key = params.k.contiguous();
  auto value = params.v.contiguous();
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "mate GDN prefill expects q/k/v shaped [B, T, H, D]");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type(),
              "mate GDN prefill expects q/k/v to share dtype");

  const int64_t batch_size = query.size(0);
  const int64_t seq_len = query.size(1);
  const int64_t pad_size = chunk_pad_size(seq_len, kGdnChunkSize);
  if (pad_size > 0) {
    query = pad_time_dim_4d(query, pad_size);
    key = pad_time_dim_4d(key, pad_size);
    value = pad_time_dim_4d(value, pad_size);
  }
  const int64_t num_tokens = query.size(1);
  const int64_t num_q_heads = query.size(2);
  const int64_t num_v_heads = value.size(2);
  const int64_t head_k_dim = query.size(3);
  const int64_t head_v_dim = value.size(3);
  TORCH_CHECK(head_k_dim == head_v_dim,
              "mate GDN prefill currently requires K == V, got K=",
              head_k_dim,
              " V=",
              head_v_dim);
  TORCH_CHECK(num_v_heads % num_q_heads == 0,
              "mate GDN prefill expects Hv divisible by Hqk");

  if (params.use_qk_l2norm_in_kernel) {
    l2norm_last_dim(query);
    l2norm_last_dim(key);
  }
  query = query.contiguous();
  key = key.contiguous();
  value = value.contiguous();

  auto beta = params.beta.to(torch::kFloat32).contiguous();
  if (pad_size > 0) {
    beta = pad_time_dim_3d(beta, pad_size, 0.0);
  }
  auto g_cumsum = chunk_local_cumsum_log_alpha(
      pad_size > 0 ? pad_time_dim_3d(params.g.to(torch::kFloat32).contiguous(),
                                     pad_size,
                                     0.0)
                   : params.g.to(torch::kFloat32).contiguous(),
      kGdnChunkSize);

  const std::string uri =
      get_mate_gdn_prefill_uri(num_q_heads, num_v_heads, query.scalar_type());
  bind_tvmffi_stream_to_current_torch_stream(query.device());
  auto run = get_function(uri, "run");

  auto a_dummy = torch::empty(
      {batch_size, num_tokens, num_v_heads, kGdnChunkSize}, query.options());
  auto h0 = torch::zeros({batch_size, num_v_heads, head_v_dim, head_k_dim},
                         torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(query.device()));
  auto output = torch::empty({batch_size, num_tokens, num_v_heads, head_v_dim},
                             value.options());
  auto final_state = torch::empty(
      {batch_size, num_v_heads, head_v_dim, head_k_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

  run(to_ffi_tensor(query),
      to_ffi_tensor(key),
      to_ffi_tensor(value),
      to_ffi_tensor(a_dummy),
      to_ffi_tensor(g_cumsum),
      to_ffi_tensor(beta),
      to_ffi_tensor(h0),
      to_ffi_tensor(output),
      to_ffi_tensor(final_state));

  if (pad_size > 0) {
    output = output.slice(/*dim=*/1, /*start=*/0, /*end=*/seq_len);
  }
  return {output, final_state};
}

}  // namespace xllm::kernel::cuda
