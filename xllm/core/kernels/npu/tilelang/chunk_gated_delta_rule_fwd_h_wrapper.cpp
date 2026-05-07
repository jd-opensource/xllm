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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <limits>
#include <tuple>

#include "acl/acl.h"
#include "dispatch_registry.h"
#include "tilelang_ops_api.h"

#ifndef XLLM_TL_CHUNK_GATED_DELTA_RULE_FWD_H_REGISTRY_INC
#error "XLLM_TL_CHUNK_GATED_DELTA_RULE_FWD_H_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_CHUNK_GATED_DELTA_RULE_FWD_H_REGISTRY_INC

void check_supported(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& initial_state,
    const torch::Tensor& cu_seqlens,
    bool use_g) {
  CHECK(k.defined()) << "TileLang ChunkGatedDeltaRuleFwdH: k must be defined";
  CHECK(w.defined()) << "TileLang ChunkGatedDeltaRuleFwdH: w must be defined";
  CHECK(v.defined()) << "TileLang ChunkGatedDeltaRuleFwdH: v must be defined";
  CHECK(cu_seqlens.defined()) << "TileLang ChunkGatedDeltaRuleFwdH: cu_seqlens must be defined";

  CHECK(k.device().type() == c10::DeviceType::PrivateUse1 &&
        w.device().type() == c10::DeviceType::PrivateUse1 &&
        v.device().type() == c10::DeviceType::PrivateUse1 &&
        cu_seqlens.device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang ChunkGatedDeltaRuleFwdH: all tensors must be on NPU";

  if (use_g) {
    CHECK(g.defined()) << "TileLang ChunkGatedDeltaRuleFwdH: g must be defined when use_g is true";
    CHECK(g.device().type() == c10::DeviceType::PrivateUse1)
        << "TileLang ChunkGatedDeltaRuleFwdH: g must be on NPU when use_g is true";
  }

  if (initial_state.defined()) {
    CHECK(initial_state.device().type() == c10::DeviceType::PrivateUse1)
        << "TileLang ChunkGatedDeltaRuleFwdH: initial_state must be on NPU if defined";
  }

  CHECK_EQ(k.dim(), 4) << "TileLang ChunkGatedDeltaRuleFwdH: k must be 4D [1, T, Hg, K]";
  CHECK_EQ(w.dim(), 4) << "TileLang ChunkGatedDeltaRuleFwdH: w must be 4D [1, T, H, K]";
  CHECK_EQ(v.dim(), 4) << "TileLang ChunkGatedDeltaRuleFwdH: v must be 4D [1, T, H, V]";
  CHECK_EQ(cu_seqlens.dim(), 1) << "TileLang ChunkGatedDeltaRuleFwdH: cu_seqlens must be 1D [N+1]";

  CHECK_EQ(k.size(0), 1) << "TileLang ChunkGatedDeltaRuleFwdH: k batch dim must be 1";
  CHECK_EQ(w.size(0), 1) << "TileLang ChunkGatedDeltaRuleFwdH: w batch dim must be 1";
  CHECK_EQ(v.size(0), 1) << "TileLang ChunkGatedDeltaRuleFwdH: v batch dim must be 1";

  CHECK_EQ(k.size(1), w.size(1))
      << "TileLang ChunkGatedDeltaRuleFwdH: k and w must have same T dim";
  CHECK_EQ(k.size(1), v.size(1))
      << "TileLang ChunkGatedDeltaRuleFwdH: k and v must have same T dim";

  CHECK(k.is_contiguous()) << "TileLang ChunkGatedDeltaRuleFwdH: k must be contiguous";
  CHECK(w.is_contiguous()) << "TileLang ChunkGatedDeltaRuleFwdH: w must be contiguous";
  CHECK(v.is_contiguous()) << "TileLang ChunkGatedDeltaRuleFwdH: v must be contiguous";

  if (use_g) {
    CHECK_EQ(g.dim(), 3) << "TileLang ChunkGatedDeltaRuleFwdH: g must be 3D [1, T, H]";
    CHECK_EQ(g.size(0), 1) << "TileLang ChunkGatedDeltaRuleFwdH: g batch dim must be 1";
    CHECK_EQ(g.size(1), k.size(1))
        << "TileLang ChunkGatedDeltaRuleFwdH: g and k must have same T dim";
    CHECK_EQ(g.size(2), w.size(2))
        << "TileLang ChunkGatedDeltaRuleFwdH: g and w must have same H dim";
    CHECK(g.is_contiguous()) << "TileLang ChunkGatedDeltaRuleFwdH: g must be contiguous";
  }

  if (initial_state.defined()) {
    CHECK_EQ(initial_state.dim(), 5)
        << "TileLang ChunkGatedDeltaRuleFwdH: initial_state must be 5D [1, N, H, K, V]";
    CHECK_EQ(initial_state.size(0), 1)
        << "TileLang ChunkGatedDeltaRuleFwdH: initial_state batch dim must be 1";
    CHECK_EQ(initial_state.size(2), w.size(2))
        << "TileLang ChunkGatedDeltaRuleFwdH: initial_state and w must have same H dim";
    CHECK_EQ(initial_state.size(3), k.size(3))
        << "TileLang ChunkGatedDeltaRuleFwdH: initial_state and k must have same K dim";
    CHECK_EQ(initial_state.size(4), v.size(3))
        << "TileLang ChunkGatedDeltaRuleFwdH: initial_state and v must have same V dim";
  }

  CHECK_GT(cu_seqlens.size(0), 1)
      << "TileLang ChunkGatedDeltaRuleFwdH: cu_seqlens must have at least 2 elements";
  CHECK_EQ(cu_seqlens[cu_seqlens.size(0) - 1].item<int64_t>(), k.size(1))
      << "TileLang ChunkGatedDeltaRuleFwdH: cu_seqlens last element must match input T_total";

  CHECK_EQ(w.size(2) % k.size(2), 0)
      << "TileLang ChunkGatedDeltaRuleFwdH: H must be divisible by Hg for GQA";
  CHECK_EQ(k.size(3) % 2, 0)
      << "TileLang ChunkGatedDeltaRuleFwdH: K must be even";
  CHECK_EQ(v.size(3) % 2, 0)
      << "TileLang ChunkGatedDeltaRuleFwdH: V must be even";
}

ChunkGatedDeltaRuleFwdHSpecialization build_runtime_specialization(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& v,
    int32_t chunk_size,
    bool use_g) {
  CHECK_EQ(k.dim(), 4) << "TileLang ChunkGatedDeltaRuleFwdH: k must be 4D";
  CHECK_EQ(w.dim(), 4) << "TileLang ChunkGatedDeltaRuleFwdH: w must be 4D";
  CHECK_EQ(v.dim(), 4) << "TileLang ChunkGatedDeltaRuleFwdH: v must be 4D";

  const int32_t H = static_cast<int32_t>(w.size(2));
  const int32_t Hg = static_cast<int32_t>(k.size(2));
  const int32_t K = static_cast<int32_t>(k.size(3));
  const int32_t V = static_cast<int32_t>(v.size(3));
  const int32_t BT = chunk_size;
  const int32_t USE_G = use_g ? 1 : 0;

  return make_chunk_gated_delta_rule_fwd_h_specialization(
      ChunkGatedDeltaRuleFwdHH{H},
      ChunkGatedDeltaRuleFwdHHg{Hg},
      ChunkGatedDeltaRuleFwdHK{K},
      ChunkGatedDeltaRuleFwdHV{V},
      ChunkGatedDeltaRuleFwdHBT{BT},
      ChunkGatedDeltaRuleFwdHUSEG{USE_G},
      ChunkGatedDeltaRuleFwdHDType{to_tilelang_dtype(k.scalar_type())});
}

torch::Tensor prepare_chunk_offsets_cpu(const torch::Tensor& cu_seqlens, int32_t chunk_size) {
  const int64_t N = cu_seqlens.size(0) - 1;
  torch::Tensor chunk_offsets = torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  int64_t offset = 0;
  for (int64_t i = 0; i < N; ++i) {
    chunk_offsets[i] = offset;
    const int64_t T_len = cu_seqlens[i + 1].item<int64_t>() - cu_seqlens[i].item<int64_t>();
    const int64_t NT = (T_len + chunk_size - 1) / chunk_size;
    offset += NT;
  }
  return chunk_offsets;
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> chunk_gated_delta_rule_fwd_h(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool save_new_value,
    const torch::Tensor& cu_seqlens) {
  const bool use_g = g.defined();
  check_supported(k, w, v, g, initial_state, cu_seqlens, use_g);

  const int64_t T_total = k.size(1);
  const int64_t H = w.size(2);
  const int64_t Hg = k.size(2);
  const int64_t K = k.size(3);
  const int64_t V = v.size(3);
  const int64_t N = cu_seqlens.size(0) - 1;

  CHECK_GT(N, 0) << "TileLang ChunkGatedDeltaRuleFwdH: N must be > 0";
  CHECK_LE(N, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "TileLang ChunkGatedDeltaRuleFwdH: N exceeds int32 range";

  const int32_t BT = static_cast<int32_t>(chunk_size);

  int32_t NT_max = 0;
  int64_t NT_total = 0;
  for (int64_t i = 0; i < N; ++i) {
    const int64_t T_len = cu_seqlens[i + 1].item<int64_t>() - cu_seqlens[i].item<int64_t>();
    const int64_t NT = (T_len + chunk_size - 1) / chunk_size;
    NT_max = std::max(NT_max, static_cast<int32_t>(NT));
    NT_total += NT;
  }

  const ChunkGatedDeltaRuleFwdHSpecialization specialization =
      build_runtime_specialization(k, w, v, BT, use_g);
  const auto* entry = find_chunk_gated_delta_rule_fwd_h_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang ChunkGatedDeltaRuleFwdH: no compiled variant. Available variants: "
      << available_chunk_gated_delta_rule_fwd_h_variant_keys();

  const torch::Tensor k_flat = k.squeeze(0);
  const torch::Tensor w_flat = w.squeeze(0);
  const torch::Tensor v_flat = v.squeeze(0);
  torch::Tensor g_flat;
  if (use_g) {
    g_flat = g.squeeze(0).to(torch::kFloat32).transpose(0, 1).contiguous();
  } else {
    g_flat = torch::empty({H, T_total}, torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
  }

  torch::Tensor v_new_flat = torch::zeros({T_total, H, V}, torch::TensorOptions().dtype(k.scalar_type()).device(k.device()));
  torch::Tensor h_out = torch::zeros({N, NT_max, H, K, V}, torch::TensorOptions().dtype(k.scalar_type()).device(k.device()));
  torch::Tensor h0 = torch::zeros({N, H, K, V}, torch::TensorOptions().dtype(k.scalar_type()).device(k.device()));
  if (initial_state.defined()) {
    h0.copy_(initial_state.squeeze(0));
  }

  torch::Tensor ht = torch::zeros({N, H, K, V}, torch::TensorOptions().dtype(k.scalar_type()).device(k.device()));

  const int32_t V_half = V / 2;
  torch::Tensor ws_wh = torch::zeros({N, H, 2, BT, V_half}, torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
  torch::Tensor ws_vnew = torch::zeros({N, H, 2, BT, V_half}, torch::TensorOptions().dtype(k.scalar_type()).device(k.device()));
  torch::Tensor ws_hupd = torch::zeros({N, H, 2, K, V_half}, torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
  torch::Tensor ws_h = torch::zeros({N, H, 2, K, V_half}, torch::TensorOptions().dtype(k.scalar_type()).device(k.device()));

  const torch::Tensor cu_seqlens_int32 = cu_seqlens.to(torch::kInt32);

  const int32_t device_id = k.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  const int32_t N_int32 = static_cast<int32_t>(N);
  const int32_t T_total_int32 = static_cast<int32_t>(T_total);
  const int32_t STORE_FINAL_STATE = output_final_state ? 1 : 0;
  const int32_t SAVE_NEW_VALUE_INT32 = save_new_value ? 1 : 0;

  entry->fn(
      /*h=*/reinterpret_cast<uint8_t*>(h_out.data_ptr()),
      /*k=*/reinterpret_cast<uint8_t*>(const_cast<void*>(k_flat.data_ptr())),
      /*v=*/reinterpret_cast<uint8_t*>(const_cast<void*>(v_flat.data_ptr())),
      /*w=*/reinterpret_cast<uint8_t*>(const_cast<void*>(w_flat.data_ptr())),
      /*g=*/reinterpret_cast<uint8_t*>(const_cast<void*>(g_flat.data_ptr())),
      /*v_new=*/reinterpret_cast<uint8_t*>(v_new_flat.data_ptr()),
      /*h0=*/reinterpret_cast<uint8_t*>(const_cast<void*>(h0.data_ptr())),
      /*ht=*/reinterpret_cast<uint8_t*>(ht.data_ptr()),
      /*cu_seqlens=*/reinterpret_cast<uint8_t*>(const_cast<void*>(cu_seqlens_int32.data_ptr())),
      /*ws_wh=*/reinterpret_cast<uint8_t*>(ws_wh.data_ptr()),
      /*ws_vnew=*/reinterpret_cast<uint8_t*>(ws_vnew.data_ptr()),
      /*ws_hupd=*/reinterpret_cast<uint8_t*>(ws_hupd.data_ptr()),
      /*ws_h=*/reinterpret_cast<uint8_t*>(ws_h.data_ptr()),
      /*N=*/N_int32,
      /*T_total=*/T_total_int32,
      /*NT_max=*/NT_max,
      /*STORE_FINAL_STATE=*/STORE_FINAL_STATE,
      /*SAVE_NEW_VALUE=*/SAVE_NEW_VALUE_INT32,
      /*stream=*/stream);

  torch::Tensor v_new = v_new_flat.unsqueeze(0);

  torch::Tensor chunk_offsets = prepare_chunk_offsets_cpu(cu_seqlens, chunk_size).to(k.device());
  torch::Tensor h = torch::zeros({NT_total, H, K, V}, torch::TensorOptions().dtype(k.scalar_type()).device(k.device()));
  for (int64_t i = 0; i < N; ++i) {
    const int64_t T_len = cu_seqlens[i + 1].item<int64_t>() - cu_seqlens[i].item<int64_t>();
    const int64_t NT_i = (T_len + chunk_size - 1) / chunk_size;
    const int64_t offset = chunk_offsets[i].item<int64_t>();
    h.slice(0, offset, offset + NT_i) = h_out.slice(1, 0, NT_i)[i];
  }
  h = h.unsqueeze(0);

  torch::Tensor ht_out;
  if (output_final_state) {
    ht_out = ht;
  }

  return std::make_tuple(h, v_new, ht_out);
}

}  // namespace xllm::kernel::npu::tilelang