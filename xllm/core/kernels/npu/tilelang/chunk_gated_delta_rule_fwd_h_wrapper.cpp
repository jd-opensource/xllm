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
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_CHUNK_GATED_DELTA_RULE_FWD_H_REGISTRY_INC
#error "XLLM_TL_CHUNK_GATED_DELTA_RULE_FWD_H_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kCompileBT = 64;
constexpr int32_t kNSpecializationMin = 1;
constexpr int32_t kNSpecializationStep = 1;

bool is_supported_initial_state_dtype(torch::ScalarType dtype) {
  return dtype == torch::kBFloat16 || dtype == torch::kFloat32;
}

#include XLLM_TL_CHUNK_GATED_DELTA_RULE_FWD_H_REGISTRY_INC

int32_t max_compiled_n(int32_t H,
                       int32_t Hg,
                       int32_t K,
                       int32_t V,
                       TilelangDType dtype) {
  int32_t max_n = 0;
  for (const auto& entry : kChunkGatedDeltaRuleFwdHRegistry) {
    const auto& spec = entry.spec;
    if (spec.H == H && spec.Hg == Hg && spec.K == K && spec.V == V &&
        spec.dtype == dtype) {
      max_n = std::max(max_n, spec.N);
    }
  }
  return max_n;
}

int32_t select_launch_n(int64_t actual_n,
                        int32_t H,
                        int32_t Hg,
                        int32_t K,
                        int32_t V,
                        TilelangDType dtype) {
  CHECK_GT(actual_n, 0)
      << "TileLang chunk_gated_delta_rule_fwd_h: actual_n must be > 0";
  const int32_t max_n = max_compiled_n(H, Hg, K, V, dtype);
  CHECK_GT(max_n, 0)
      << "TileLang chunk_gated_delta_rule_fwd_h: no compiled N variant for "
      << "H=" << H << ", Hg=" << Hg << ", K=" << K << ", V=" << V
      << ", dtype=" << static_cast<int>(dtype);
  CHECK_GE(max_n, kNSpecializationMin)
      << "TileLang chunk_gated_delta_rule_fwd_h: compiled N variants must "
      << "be >= " << kNSpecializationMin;

  const int64_t capped = std::min<int64_t>(actual_n, max_n);
  int64_t rounded_up =
      ((capped + kNSpecializationStep - 1) / kNSpecializationStep) *
      kNSpecializationStep;
  rounded_up = std::max<int64_t>(rounded_up, kNSpecializationMin);
  rounded_up = std::min<int64_t>(rounded_up, max_n);
  return static_cast<int32_t>(rounded_up);
}

void check_supported(const torch::Tensor& k,
                     const torch::Tensor& w,
                     const torch::Tensor& u,
                     const std::optional<torch::Tensor>& g,
                     const std::optional<torch::Tensor>& initial_state,
                     int64_t chunk_size,
                     const std::optional<torch::Tensor>& cu_seqlens) {
  CHECK(k.defined())
      << "TileLang chunk_gated_delta_rule_fwd_h: k must be defined";
  CHECK(w.defined())
      << "TileLang chunk_gated_delta_rule_fwd_h: w must be defined";
  CHECK(u.defined())
      << "TileLang chunk_gated_delta_rule_fwd_h: u must be defined";
  CHECK(g.has_value())
      << "TileLang chunk_gated_delta_rule_fwd_h: g must be defined";

  CHECK(k.device().type() == c10::DeviceType::PrivateUse1 &&
        w.device().type() == c10::DeviceType::PrivateUse1 &&
        u.device().type() == c10::DeviceType::PrivateUse1 &&
        g.value().device().type() == c10::DeviceType::PrivateUse1)
      << "TileLang chunk_gated_delta_rule_fwd_h: all tensors must be on NPU";

  CHECK_EQ(k.dim(), 3)
      << "TileLang chunk_gated_delta_rule_fwd_h: k must be 3D [T, Hg, K]";
  CHECK_EQ(w.dim(), 3)
      << "TileLang chunk_gated_delta_rule_fwd_h: w must be 3D [T, H, K]";
  CHECK_EQ(u.dim(), 3)
      << "TileLang chunk_gated_delta_rule_fwd_h: u must be 3D [T, H, V]";

  CHECK_EQ(k.size(0), w.size(0))
      << "TileLang chunk_gated_delta_rule_fwd_h: k/w token dim mismatch";
  CHECK_EQ(k.size(0), u.size(0))
      << "TileLang chunk_gated_delta_rule_fwd_h: k/u token dim mismatch";
  CHECK_EQ(w.size(1), u.size(1))
      << "TileLang chunk_gated_delta_rule_fwd_h: w/u head dim mismatch";

  CHECK_EQ(k.dtype(), torch::kBFloat16)
      << "TileLang chunk_gated_delta_rule_fwd_h: only bf16 inputs are "
         "supported";
  CHECK_EQ(k.dtype(), w.dtype())
      << "TileLang chunk_gated_delta_rule_fwd_h: k/w dtype mismatch";
  CHECK_EQ(k.dtype(), u.dtype())
      << "TileLang chunk_gated_delta_rule_fwd_h: k/u dtype mismatch";

  if (initial_state.has_value()) {
    CHECK(is_supported_initial_state_dtype(initial_state.value().scalar_type()))
        << "TileLang chunk_gated_delta_rule_fwd_h: initial_state must be "
        << "bfloat16 or float32, got " << initial_state.value().scalar_type();
  }

  CHECK_EQ(chunk_size, kCompileBT)
      << "TileLang chunk_gated_delta_rule_fwd_h: only chunk_size=" << kCompileBT
      << " is supported, got " << chunk_size;

  CHECK(cu_seqlens.has_value())
      << "TileLang chunk_gated_delta_rule_fwd_h: cu_seqlens must be defined";

  const int64_t N = cu_seqlens.value().size(0) - 1;
  CHECK_GT(N, 0) << "TileLang chunk_gated_delta_rule_fwd_h: N must be > 0";
}

ChunkGatedDeltaRuleFwdHSpecialization build_runtime_specialization(
    int32_t compiled_n,
    const torch::Tensor& k,
    const torch::Tensor& u) {
  CHECK_EQ(k.dim(), 3);
  CHECK_EQ(u.dim(), 3);
  const int32_t H = static_cast<int32_t>(u.size(1));
  const int32_t Hg = static_cast<int32_t>(k.size(1));
  const int32_t K = static_cast<int32_t>(k.size(2));
  const int32_t V = static_cast<int32_t>(u.size(2));
  const TilelangDType dtype = to_tilelang_dtype(k.scalar_type());

  return make_chunk_gated_delta_rule_fwd_h_specialization(
      ChunkGatedDeltaRuleFwdHN{compiled_n},
      ChunkGatedDeltaRuleFwdHH{H},
      ChunkGatedDeltaRuleFwdHHg{Hg},
      ChunkGatedDeltaRuleFwdHK{K},
      ChunkGatedDeltaRuleFwdHV{V},
      ChunkGatedDeltaRuleFwdHDType{dtype});
}

const auto* find_entry_with_fallback(
    ChunkGatedDeltaRuleFwdHSpecialization specialization) {
  const auto* entry =
      find_chunk_gated_delta_rule_fwd_h_kernel_entry(specialization);
  if (entry != nullptr) {
    return entry;
  }
  int32_t fallback_n = specialization.N - kNSpecializationStep;
  while (fallback_n >= kNSpecializationMin && entry == nullptr) {
    specialization = make_chunk_gated_delta_rule_fwd_h_specialization(
        ChunkGatedDeltaRuleFwdHN{fallback_n},
        ChunkGatedDeltaRuleFwdHH{specialization.H},
        ChunkGatedDeltaRuleFwdHHg{specialization.Hg},
        ChunkGatedDeltaRuleFwdHK{specialization.K},
        ChunkGatedDeltaRuleFwdHV{specialization.V},
        ChunkGatedDeltaRuleFwdHDType{specialization.dtype});
    entry = find_chunk_gated_delta_rule_fwd_h_kernel_entry(specialization);
    fallback_n -= kNSpecializationStep;
  }
  return entry;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
run_tilelang_chunk_gated_delta_rule_fwd_h(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& u,
    const torch::Tensor& g,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    const torch::Tensor& cu_seqlens) {
  const int64_t actual_n = cu_seqlens.size(0) - 1;
  const int64_t t_total = k.size(0);
  const int64_t H = u.size(1);
  const int64_t Hg = k.size(1);
  const int64_t K = k.size(2);
  const int64_t V = u.size(2);
  const TilelangDType dtype = to_tilelang_dtype(k.scalar_type());

  const int32_t compiled_n = select_launch_n(actual_n,
                                             static_cast<int32_t>(H),
                                             static_cast<int32_t>(Hg),
                                             static_cast<int32_t>(K),
                                             static_cast<int32_t>(V),
                                             dtype);

  int32_t actual_nt_max = 0;
  {
    auto cu_cpu = cu_seqlens.cpu().contiguous();
    auto cu_ptr = cu_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < actual_n; ++i) {
      const int32_t seq_len = cu_ptr[i + 1] - cu_ptr[i];
      const int32_t nt_i = (seq_len + kCompileBT - 1) / kCompileBT;
      actual_nt_max = std::max(actual_nt_max, nt_i);
    }
  }
  CHECK_GT(actual_nt_max, 0)
      << "TileLang chunk_gated_delta_rule_fwd_h: actual_nt_max must be > 0";

  const auto options_bf16 = k.options();
  const auto options_fp32 = k.options().dtype(torch::kFloat32);

  const int32_t V_half = static_cast<int32_t>(V) / 2;
  const int64_t t_total_i64 = static_cast<int64_t>(t_total);
  const int64_t actual_nt_i64 = static_cast<int64_t>(actual_nt_max);

  auto h_out = torch::empty({compiled_n, actual_nt_i64, H, K, V}, options_bf16);
  auto v_new_out = torch::empty({t_total_i64, H, V}, options_bf16);
  auto h0 = torch::zeros({compiled_n, H, K, V}, options_bf16);
  if (initial_state.has_value()) {
    h0.narrow(0, 0, actual_n).copy_(initial_state.value().to(torch::kBFloat16));
  }
  auto ht = torch::zeros({compiled_n, H, K, V}, options_fp32);

  auto ws_wh =
      torch::empty({compiled_n, H, 2, kCompileBT, V_half}, options_fp32);
  auto ws_vnew =
      torch::empty({compiled_n, H, 2, kCompileBT, V_half}, options_bf16);
  auto ws_hupd = torch::empty({compiled_n, H, 2, K, V_half}, options_fp32);
  auto ws_h = torch::empty({compiled_n, H, 2, K, V_half}, options_bf16);

  auto cu_prepared = cu_seqlens.to(torch::kInt32).contiguous();
  if (static_cast<int64_t>(compiled_n) > actual_n) {
    auto cu_padded = torch::empty({compiled_n + 1}, cu_prepared.options());
    cu_padded.narrow(0, 0, actual_n + 1).copy_(cu_prepared);
    cu_padded.narrow(0, actual_n + 1, compiled_n - actual_n)
        .fill_(static_cast<int32_t>(t_total));
    cu_prepared = cu_padded;
  }

  auto specialization = build_runtime_specialization(compiled_n, k, u);
  const auto* entry = find_entry_with_fallback(specialization);
  CHECK(entry != nullptr)
      << "TileLang chunk_gated_delta_rule_fwd_h: no compiled variant. "
      << "Available variants: "
      << available_chunk_gated_delta_rule_fwd_h_variant_keys();

  const int32_t device_id = k.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(static_cast<uint8_t*>(h_out.data_ptr()),
            static_cast<uint8_t*>(k.data_ptr()),
            static_cast<uint8_t*>(u.data_ptr()),
            static_cast<uint8_t*>(w.data_ptr()),
            static_cast<uint8_t*>(g.data_ptr()),
            static_cast<uint8_t*>(v_new_out.data_ptr()),
            static_cast<uint8_t*>(h0.data_ptr()),
            static_cast<uint8_t*>(ht.data_ptr()),
            static_cast<uint8_t*>(cu_prepared.data_ptr()),
            static_cast<uint8_t*>(ws_wh.data_ptr()),
            static_cast<uint8_t*>(ws_vnew.data_ptr()),
            static_cast<uint8_t*>(ws_hupd.data_ptr()),
            static_cast<uint8_t*>(ws_h.data_ptr()),
            static_cast<int64_t>(actual_nt_max),
            static_cast<int64_t>(t_total),
            stream);

  auto h_sliced = h_out.narrow(0, 0, actual_n);
  auto ht_sliced =
      output_final_state ? ht.narrow(0, 0, actual_n) : torch::Tensor();

  return {h_sliced, v_new_out, ht_sliced};
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
chunk_gated_delta_rule_fwd_h(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& u,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool save_new_value,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& chunk_offsets) {
  (void)save_new_value;
  (void)chunk_offsets;
  check_supported(k, w, u, g, initial_state, chunk_size, cu_seqlens);

  auto g_c_t = g.value().to(torch::kFloat32).contiguous();
  auto cu_valid = cu_seqlens.value().contiguous();

  return run_tilelang_chunk_gated_delta_rule_fwd_h(
      k, w, u, g_c_t, initial_state, output_final_state, cu_valid);
}

}  // namespace xllm::kernel::npu::tilelang
