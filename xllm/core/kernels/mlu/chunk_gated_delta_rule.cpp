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

#include "chunk_gated_delta_rule.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <framework/core/device.h>

#include <cmath>
#include <unordered_map>

#include "chunk_fwd_o.h"
#include "chunk_gated_delta_rule_fwd_h.h"
#include "chunk_local_cumsum.h"
#include "chunk_scaled_dot_kkt_fwd.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/mlu/utils.h"
#include "recompute_w_u_fwd.h"

namespace xllm {
namespace kernel {
namespace mlu {

ChunkGatedDeltaRuleImpl::ChunkGatedDeltaRuleImpl(int64_t num_k_heads,
                                                 int64_t num_v_heads)
    : total_core_num_(0), num_k_heads_(num_k_heads), num_v_heads_(num_v_heads) {
  auto device = torch_mlu::current_device();
  auto device_prop = torch_mlu::getDeviceProperties(device);
  total_core_num_ =
      device_prop->cluster_count * device_prop->core_num_per_cluster;
  static const std::unordered_map<int64_t, int32_t> kNumVHeadToIdx = {
      {1, 0},
      {2, 1},
      {4, 2},
      {6, 3},
      {8, 4},
      {12, 5},
      {16, 6},
      {24, 7},
      {32, 8},
      {48, 9},
      {64, 10},
  };
  static const std::unordered_map<int64_t, int32_t> kNumKHeadToIdx = {
      {1, 0},
      {2, 11},
      {4, 22},
      {8, 33},
      {16, 44},
      {32, 55},
  };
  algo_id_ = lookup_algo_id(kNumKHeadToIdx,
                            num_k_heads_,
                            /*dim_name=*/"num_k_heads") +
             lookup_algo_id(kNumVHeadToIdx,
                            num_v_heads_,
                            /*dim_name=*/"num_v_heads");
  chunk_algo_id_ = lookup_algo_id(kNumVHeadToIdx,
                                  num_v_heads_,
                                  /*dim_name=*/"num_v_heads");
}

std::tuple<torch::Tensor, torch::Tensor> ChunkGatedDeltaRuleImpl::forward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    torch::Tensor& beta,
    torch::Tensor& initial_state,
    torch::Tensor& cu_seqlens,
    torch::Tensor& chunk_indices,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel) {
  // All input tensors should be contiguous
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  g = g.contiguous().to(torch::kFloat32);
  beta = beta.contiguous();
  initial_state = initial_state.contiguous();
  cu_seqlens = cu_seqlens.contiguous().to(torch::kInt32);

  if (use_qk_l2norm_in_kernel) {
    q = l2norm(q, -1);
    k = l2norm(k, -1);
  }
  float scale = 1.0f / std::sqrt(static_cast<float>(k.size(-1)));
  constexpr int64_t chunk_size = kDefaultChunkSize;

  // Step 1: Chunk preparation
  torch::Tensor g_cumsum = chunk_local_cumsum(
      g, chunk_size, false, cu_seqlens, false, chunk_indices);

  // Step 2: Permute g for downstream operations: [B, T, H] -> [H, B, T]
  torch::Tensor g_trans = g_cumsum.permute({2, 0, 1}).contiguous();

  // Step 3: Compute attention matrix A
  torch::Tensor A = chunk_scaled_dot_kkt_fwd(
      k, beta, g_cumsum, cu_seqlens, chunk_size, chunk_indices);

  // Step 4: Solve triangular system
  torch::Tensor A_tril = solve_tril(A, cu_seqlens, torch::kBFloat16);

  // Step 5: Recompute w and u
  auto [w, u] =
      recompute_w_fwd(k, v, beta, g_trans, A_tril, cu_seqlens, chunk_indices);

  // Step 6: Compute h and v_new (and final_state if needed)
  auto [h, v_new, final_state] =
      chunk_gated_delta_rule_fwd_h(k,
                                   w,
                                   u,
                                   g_trans,
                                   std::nullopt,
                                   initial_state,
                                   output_final_state,
                                   chunk_size,
                                   true,
                                   cu_seqlens,
                                   chunk_indices);

  // Step 7: Compute final output
  torch::Tensor o = chunk_fwd_o(
      q, k, v_new, h, g_trans, scale, cu_seqlens, chunk_size, chunk_indices);

  return std::make_tuple(o, final_state);
}

// ============================================================================
// Helper Methods
// ============================================================================

cnrtDim3_t ChunkGatedDeltaRuleImpl::compute_grid_dim(
    int64_t total_chunks) const {
  int64_t grid_num = std::min(total_core_num_, total_chunks);
  return cnrtDim3_t{static_cast<uint32_t>(grid_num), 1, 1};
}

torch::Tensor ChunkGatedDeltaRuleImpl::prepare_lens(
    const torch::Tensor& cu_seqlens) const {
  return cu_seqlens.narrow(0, 1, cu_seqlens.size(0) - 1) -
         cu_seqlens.narrow(0, 0, cu_seqlens.size(0) - 1);
}

torch::Tensor ChunkGatedDeltaRuleImpl::prepare_chunk_offsets(
    const torch::Tensor& cu_seqlens,
    int64_t chunk_size) const {
  torch::Tensor lengths = prepare_lens(cu_seqlens);
  torch::Tensor num_chunks = (lengths + chunk_size - 1) / chunk_size;
  num_chunks = num_chunks.to(torch::kLong);
  torch::Tensor zero = cu_seqlens.new_zeros({1});
  torch::Tensor combined = torch::cat({zero, num_chunks}, 0);
  return combined.cumsum(-1).to(torch::kInt32);
}

torch::Tensor ChunkGatedDeltaRuleImpl::prepare_chunk_indices(
    const torch::Tensor& cu_seqlens,
    int64_t chunk_size) const {
  torch::Tensor lengths = prepare_lens(cu_seqlens);
  torch::Tensor num_chunks = (lengths + chunk_size - 1) / chunk_size;
  num_chunks = num_chunks.to(torch::kLong);
  torch::Tensor cumsum = torch::cumsum(num_chunks, 0);
  int64_t total = cumsum[-1].item<int64_t>();
  torch::Tensor arange_total = torch::arange(total, cu_seqlens.options());

  torch::Tensor zeros = torch::zeros({1}, cumsum.options());
  torch::Tensor prefix = torch::cat({zeros, cumsum.slice(0, 0, -1)});
  torch::Tensor repeats_prefix = torch::repeat_interleave(prefix, num_chunks);

  torch::Tensor indices = arange_total - repeats_prefix;
  torch::Tensor mask = indices == 0;
  torch::Tensor col0 = mask.cumsum(0) - 1;
  auto out = torch::stack({col0, indices}, 1).to(cu_seqlens);
  return out.to(torch::kInt32);
}

// ============================================================================
// Core Computation Methods
// ============================================================================

torch::Tensor ChunkGatedDeltaRuleImpl::chunk_local_cumsum(
    const torch::Tensor& g,
    int64_t chunk_size,
    bool reverse,
    const torch::Tensor& cu_seqlens,
    bool head_first,
    const torch::Tensor& chunk_indices) {
  int64_t B = g.size(0);
  int64_t T = head_first ? g.size(2) : g.size(1);
  int64_t NT;
  if (cu_seqlens.numel() == 0) {
    NT = ceil_div(T, chunk_size);
  } else {
    NT = chunk_indices.size(0);
  }
  int64_t chunk_num = B * NT;

  auto output =
      torch::empty_like(g, torch::dtype(torch::kFloat32).device(g.device()));
  auto queue = torch_mlu::getCurMLUStream();
  auto dim = compute_grid_dim(chunk_num);

  tmo_chunk_local_cumsum_scalar_kernel(queue,
                                       &dim,
                                       g.data_ptr(),
                                       output.data_ptr(),
                                       cu_seqlens.data_ptr(),
                                       chunk_indices.data_ptr(),
                                       static_cast<int32_t>(T),
                                       static_cast<int32_t>(B),
                                       static_cast<int32_t>(NT),
                                       static_cast<int32_t>(chunk_num),
                                       chunk_algo_id_);
  return output;
}

torch::Tensor ChunkGatedDeltaRuleImpl::chunk_scaled_dot_kkt_fwd(
    const torch::Tensor& k,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    const torch::Tensor& cu_seqlens,
    int64_t chunk_size,
    const torch::Tensor& chunk_indices) {
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t H = beta.size(-1);
  int64_t NT;
  if (cu_seqlens.numel() == 0) {
    NT = ceil_div(T, chunk_size);
  } else {
    NT = chunk_indices.size(0);
  }
  int64_t chunk_num = B * H * NT;

  auto A = torch::empty({B, T, H, chunk_size},
                        torch::dtype(torch::kFloat32).device(k.device()));
  auto queue = torch_mlu::getCurMLUStream();
  auto dim = compute_grid_dim(chunk_num);

  tmo_chunk_scaled_dot_kkt_fwd_kernel(queue,
                                      &dim,
                                      k.data_ptr(),
                                      beta.data_ptr(),
                                      g.data_ptr(),
                                      A.data_ptr(),
                                      cu_seqlens.data_ptr(),
                                      chunk_indices.data_ptr(),
                                      static_cast<int32_t>(T),
                                      static_cast<int32_t>(B),
                                      static_cast<int32_t>(NT),
                                      static_cast<int32_t>(chunk_num),
                                      algo_id_);

  return A;
}

torch::Tensor ChunkGatedDeltaRuleImpl::solve_tril(
    const torch::Tensor& A,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::ScalarType>& output_dtype) {
  torch::ScalarType dtype =
      output_dtype.has_value() ? output_dtype.value() : A.scalar_type();

  torch::Tensor A_float =
      (A.scalar_type() != at::kFloat) ? A.to(at::kFloat) : A;

  torch::Tensor output = torch::empty(
      A_float.sizes(),
      torch::TensorOptions().dtype(dtype).device(A_float.device()));

  if (cu_seqlens.has_value()) {
    tmo::torch_api::solve_tril(
        A_float, output, cu_seqlens.value().to(torch::kInt32));
  } else {
    tmo::torch_api::solve_tril(A_float, output, std::nullopt);
  }

  return output;
}

std::pair<torch::Tensor, torch::Tensor>
ChunkGatedDeltaRuleImpl::recompute_w_fwd(const torch::Tensor& k,
                                         const torch::Tensor& v,
                                         const torch::Tensor& beta,
                                         const torch::Tensor& g_cumsum,
                                         const torch::Tensor& A,
                                         const torch::Tensor& cu_seqlens,
                                         const torch::Tensor& chunk_indices) {
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t K = k.size(3);
  int64_t H = v.size(-2);
  int64_t BT = A.size(-1);
  int64_t NT;
  if (cu_seqlens.numel() == 0) {
    NT = ceil_div(T, BT);
  } else {
    NT = chunk_indices.size(0);
  }
  int64_t chunk_num = B * H * NT;

  auto u = torch::empty_like(v, k.device());
  auto w = k.new_empty({B, T, H, K}, k.device());
  auto trans_beta = beta.permute({2, 0, 1}).contiguous();

  auto queue = torch_mlu::getCurMLUStream();
  auto dim = compute_grid_dim(chunk_num);

  tmo_recompute_w_u_fwd_kernel(queue,
                               &dim,
                               k.data_ptr(),
                               v.data_ptr(),
                               trans_beta.data_ptr(),
                               w.data_ptr(),
                               u.data_ptr(),
                               A.data_ptr(),
                               get_ptr_or_null(g_cumsum),
                               get_ptr_or_null(cu_seqlens),
                               get_ptr_or_null(chunk_indices),
                               static_cast<int32_t>(T),
                               static_cast<int32_t>(B),
                               static_cast<int32_t>(NT),
                               static_cast<int32_t>(chunk_num),
                               algo_id_);

  return std::make_pair(w, u);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ChunkGatedDeltaRuleImpl::chunk_gated_delta_rule_fwd_h(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& u,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool save_new_value,
    const std::optional<torch::Tensor>& cu_seqlens,
    const torch::Tensor& chunk_indices) {
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t K = k.size(3);
  int64_t V = u.size(-1);
  int64_t H = u.size(-2);

  int64_t N, NT;
  torch::Tensor cu_seqlens_tensor, chunk_offsets;

  if (!cu_seqlens.has_value()) {
    N = B;
    NT = ceil_div(T, chunk_size);
  } else {
    cu_seqlens_tensor = cu_seqlens.value();
    N = cu_seqlens_tensor.size(0) - 1;
    NT = chunk_indices.size(0);
    chunk_offsets = prepare_chunk_offsets(cu_seqlens_tensor, chunk_size);
  }

  auto h = k.new_empty({B, NT, H, K, V}, k.device());
  torch::Tensor final_state, v_new;
  if (output_final_state) {
    final_state = k.new_empty({N, H, K, V}, torch::kFloat32);
  }
  if (save_new_value) {
    v_new = torch::empty_like(u);
  }

  int64_t NV = ceil_div(V, kBv);
  cnrtDim3_t dim_block = {
      static_cast<uint32_t>(NV), static_cast<uint32_t>(N * H), 1};
  auto queue = torch_mlu::getCurMLUStream();

  tmo_chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
      queue,
      &dim_block,
      k.data_ptr(),
      u.data_ptr(),
      w.data_ptr(),
      get_ptr_or_null(v_new),
      get_ptr_or_null(g),
      get_ptr_or_null(gk),
      h.data_ptr(),
      get_ptr_or_null(initial_state),
      get_ptr_or_null(final_state),
      get_ptr_or_null(cu_seqlens_tensor),
      get_ptr_or_null(chunk_offsets),
      static_cast<int32_t>(T),
      static_cast<int32_t>(B),
      algo_id_);

  return std::make_tuple(h, v_new, final_state);
}

torch::Tensor ChunkGatedDeltaRuleImpl::chunk_fwd_o(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& h,
    const std::optional<torch::Tensor>& g,
    const std::optional<float>& scale,
    const std::optional<torch::Tensor>& cu_seqlens,
    int64_t chunk_size,
    const torch::Tensor& chunk_indices) {
  int64_t B = q.size(0);
  int64_t T = q.size(1);
  int64_t H = v.size(-2);
  int64_t BT = kDefaultChunkSize;
  int64_t NT;
  if (!cu_seqlens.has_value()) {
    NT = ceil_div(T, BT);
  } else {
    NT = chunk_indices.size(0);
  }
  int64_t chunk_num = B * H * NT;

  float scale_value = scale.has_value()
                          ? scale.value()
                          : 1.0f / std::sqrt(static_cast<float>(k.size(-1)));

  torch::Tensor o = torch::empty_like(v);
  torch::Tensor g_internal = g.has_value() ? g.value() : torch::Tensor();
  torch::Tensor cu_seqlens_internal =
      cu_seqlens.has_value() ? cu_seqlens.value() : torch::Tensor();

  auto queue = torch_mlu::getCurMLUStream();
  auto dim = compute_grid_dim(chunk_num);

  tmo_chunk_fwd_kernel_o(queue,
                         &dim,
                         q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         h.data_ptr(),
                         get_ptr_or_null(g_internal),
                         o.data_ptr(),
                         get_ptr_or_null(cu_seqlens_internal),
                         get_ptr_or_null(chunk_indices),
                         scale_value,
                         static_cast<int32_t>(T),
                         static_cast<int32_t>(B),
                         static_cast<int32_t>(NT),
                         static_cast<int32_t>(chunk_num),
                         algo_id_);

  return o;
}
}  // namespace mlu
}  // namespace kernel
}  // namespace xllm
