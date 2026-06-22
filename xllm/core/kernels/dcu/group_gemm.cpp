/* Copyright 2025-2026 The xLLM Authors.

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

/// @brief DCU grouped GEMM bridge: routes xllm::kernel::dcu::group_gemm()
///        to the CK Tile grouped_gemm_tileloop kernel via the per-dtype
///        C ABI functions.
///
/// IMPORTANT: This file forward-declares the CK C ABI struct and functions
/// instead of including ck/grouped_gemm/grouped_gemm.hpp.  That header
/// pulls in the entire ck_tile template library (via ck_tile/ops/gemm.hpp),
/// which contains HIP device code that conflicts with the system clang++
/// (e.g. __fp16 vs _Float16 type mismatch in warp_dsreadm_attribute_impl.hpp,
/// and missing <fstream> in json_dump.hpp).

#include <glog/logging.h>
#include <hip/hip_runtime.h>  // for hipStream_t

#include "dcu_ops_api.h"

// ---------------------------------------------------------------------------
// Forward declarations — must match ck/grouped_gemm/grouped_gemm.hpp exactly
// ---------------------------------------------------------------------------

struct ck_tile_dcu_grouped_gemm_desc {
  const void* a_ptr;
  const void* b_ptr;
  void* c_ptr;
  int k_batch;
  int M;
  int N;
  int K;
  int stride_A;
  int stride_B;
  int stride_C;
  int num_d_tensors;
  const void* const* d_ptrs;
  const int* stride_Ds;
};

// Per-dtype C ABI functions defined in the CK instance translation units
// (instances/grouped_gemm_*.cpp).
int grouped_gemm_c_run_fp16(const ck_tile_dcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream,
                            int warmup,
                            int repeat,
                            float* avg_ms);

int grouped_gemm_c_run_bf16(const ck_tile_dcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream,
                            int warmup,
                            int repeat,
                            float* avg_ms);

int grouped_gemm_c_run_fp8(const ck_tile_dcu_grouped_gemm_desc* descs,
                           int group_count,
                           char a_layout,
                           char b_layout,
                           void* workspace,
                           hipStream_t stream,
                           int warmup,
                           int repeat,
                           float* avg_ms);

int grouped_gemm_c_run_bf8(const ck_tile_dcu_grouped_gemm_desc* descs,
                           int group_count,
                           char a_layout,
                           char b_layout,
                           void* workspace,
                           hipStream_t stream,
                           int warmup,
                           int repeat,
                           float* avg_ms);

int grouped_gemm_c_run_int8(const ck_tile_dcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream,
                            int warmup,
                            int repeat,
                            float* avg_ms);

int grouped_gemm_c_run_int4(const ck_tile_dcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream,
                            int warmup,
                            int repeat,
                            float* avg_ms);

// ---------------------------------------------------------------------------
// Workspace-size helper
// ---------------------------------------------------------------------------
// Declared extern "C" in grouped_gemm.hpp.  Defined here (instead of
// linking grouped_gemm.cpp which carries main()) with a conservative
// upper-bound estimate.  The real GemmTransKernelArg is ~64 bytes;
// 512 bytes/group provides generous headroom.
extern "C" std::size_t ck_tile_dcu_grouped_gemm_workspace_size(
    int group_count,
    int /*num_d_tensors*/) {
  if (group_count <= 0) return 0;
  constexpr std::size_t kMaxKernelArgSize = 512;
  return static_cast<std::size_t>(group_count) * kMaxKernelArgSize;
}

// ---------------------------------------------------------------------------
// Bridge implementation
// ---------------------------------------------------------------------------

namespace xllm::kernel::dcu {

torch::Tensor group_gemm(const torch::Tensor& input,
                         const torch::Tensor& weight,
                         const torch::Tensor& token_count,
                         std::optional<torch::Tensor> output_opt) {
  CHECK(input.is_contiguous()) << "input must be contiguous";
  CHECK(weight.is_contiguous()) << "weight must be contiguous";
  CHECK(token_count.is_contiguous()) << "token_count must be contiguous";
  const auto dtype = input.scalar_type();
  const int64_t N = weight.size(1);  // output feature dim
  const int64_t K = weight.size(2);  // input feature dim
  const int64_t element_size = input.element_size();
  const int num_experts = static_cast<int>(token_count.size(0));

  // Allocate output if not provided.
  torch::Tensor output;
  if (output_opt.has_value()) {
    output = output_opt.value();
  } else {
    output = torch::empty({input.size(0), N}, input.options());
  }

  // Pull token_count to host for descriptor construction.
  auto token_count_cpu = token_count.to(torch::kInt32).cpu();
  const auto* token_count_ptr = token_count_cpu.data_ptr<int32_t>();

  // Byte-addressable base pointers for offset arithmetic.
  const auto* a_base = static_cast<const char*>(input.data_ptr());
  const auto* b_base = static_cast<const char*>(weight.data_ptr());
  auto* c_base = static_cast<char*>(output.data_ptr());

  std::vector<ck_tile_dcu_grouped_gemm_desc> descs;
  descs.reserve(num_experts);

  int64_t token_offset = 0;
  for (int32_t e = 0; e < num_experts; ++e) {
    int32_t M_e = token_count_ptr[e];
    if (M_e == 0) continue;

    ck_tile_dcu_grouped_gemm_desc desc{};
    desc.a_ptr = a_base + token_offset * K * element_size;
    desc.b_ptr = b_base + static_cast<int64_t>(e) * N * K * element_size;
    desc.c_ptr = c_base + token_offset * N * element_size;
    desc.k_batch = 1;
    desc.M = M_e;
    desc.N = static_cast<int>(N);
    desc.K = static_cast<int>(K);
    desc.stride_A = static_cast<int>(K);
    desc.stride_B = static_cast<int>(K);
    desc.stride_C = static_cast<int>(N);
    desc.num_d_tensors = 0;
    desc.d_ptrs = nullptr;
    desc.stride_Ds = nullptr;
    descs.push_back(desc);

    token_offset += M_e;
  }

  if (descs.empty()) {
    return output;
  }

  const int group_count = static_cast<int>(descs.size());

  // Pass nullptr stream — the CK kernel runs on the default stream.
  // This matches how both the CK example and test code call these functions.
  float avg_ms = 0.0f;
  int ret = 0;
  if (dtype == torch::kFloat16) {
    ret = grouped_gemm_c_run_fp16(descs.data(),
                                  group_count,
                                  'R',
                                  'C',
                                  /*workspace=*/nullptr,
                                  /*stream=*/nullptr,
                                  /*warmup=*/0,
                                  /*repeat=*/1,
                                  &avg_ms);
  } else if (dtype == torch::kBFloat16) {
    ret = grouped_gemm_c_run_bf16(descs.data(),
                                  group_count,
                                  'R',
                                  'C',
                                  /*workspace=*/nullptr,
                                  /*stream=*/nullptr,
                                  /*warmup=*/0,
                                  /*repeat=*/1,
                                  &avg_ms);
  } else if (dtype == torch::kFloat8_e4m3fn) {
    ret = grouped_gemm_c_run_fp8(descs.data(),
                                 group_count,
                                 'R',
                                 'C',
                                 /*workspace=*/nullptr,
                                 /*stream=*/nullptr,
                                 /*warmup=*/0,
                                 /*repeat=*/1,
                                 &avg_ms);
  } else if (dtype == torch::kFloat8_e5m2) {
    // bf8 shares the float8_e5m2 representation in PyTorch.
    ret = grouped_gemm_c_run_bf8(descs.data(),
                                 group_count,
                                 'R',
                                 'C',
                                 /*workspace=*/nullptr,
                                 /*stream=*/nullptr,
                                 /*warmup=*/0,
                                 /*repeat=*/1,
                                 &avg_ms);
  } else if (dtype == torch::kInt8) {
    // Also covers int4: packed int4 data is stored as int8 bytes in torch.
    ret = grouped_gemm_c_run_int8(descs.data(),
                                  group_count,
                                  'R',
                                  'C',
                                  /*workspace=*/nullptr,
                                  /*stream=*/nullptr,
                                  /*warmup=*/0,
                                  /*repeat=*/1,
                                  &avg_ms);
  } else {
    LOG(FATAL) << "dcu::group_gemm: unsupported dtype " << dtype;
  }

  CHECK_EQ(ret, 0) << "dcu::group_gemm: CK kernel returned error code " << ret;

  return output;
}

}  // namespace xllm::kernel::dcu
