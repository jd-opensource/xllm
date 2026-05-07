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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "acl/acl.h"
#include "tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangChunkGatedDeltaRuleFwdHWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct ChunkGatedDeltaRuleFwdHTestCase {
  std::string name;
  std::vector<int64_t> seqlens;
  int64_t H;
  int64_t Hg;
  int64_t K;
  int64_t V;
  int64_t chunk_size;
  bool use_g;
  bool use_initial_state;
  int64_t seed;
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> torch_ref_chunk_gated_delta_rule(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& u,
    const torch::Tensor& g,
    const torch::Tensor& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    const torch::Tensor& cu_seqlens) {
  const int64_t BT = chunk_size;
  const torch::Tensor k_fp32 = k.squeeze(0).to(torch::kFloat32);
  const torch::Tensor w_fp32 = w.squeeze(0).to(torch::kFloat32);
  const torch::Tensor u_fp32 = u.squeeze(0).to(torch::kFloat32);
  torch::Tensor g_fp32;
  if (g.defined()) {
    g_fp32 = g.squeeze(0).to(torch::kFloat32);
  }
  torch::Tensor h0_fp32;
  if (initial_state.defined()) {
    h0_fp32 = initial_state.squeeze(0).to(torch::kFloat32);
  }

  const int64_t T_total = k_fp32.size(0);
  const int64_t Hg = k_fp32.size(1);
  const int64_t K = k_fp32.size(2);
  const int64_t H = w_fp32.size(1);
  const int64_t V = u_fp32.size(2);
  const int64_t N = cu_seqlens.size(0) - 1;

  int64_t NT_total = 0;
  for (int64_t i = 0; i < N; ++i) {
    const int64_t T_len = cu_seqlens[i + 1].item<int64_t>() - cu_seqlens[i].item<int64_t>();
    NT_total += (T_len + BT - 1) / BT;
  }

  torch::Tensor h = torch::zeros({NT_total, H, K, V}, torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
  torch::Tensor v_new = torch::zeros({T_total, H, V}, torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
  torch::Tensor final_state;
  if (output_final_state) {
    final_state = torch::zeros({N, H, K, V}, torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
  }

  int64_t chunk_offset = 0;
  for (int64_t i_n = 0; i_n < N; ++i_n) {
    const int64_t bos = cu_seqlens[i_n].item<int64_t>();
    const int64_t eos = cu_seqlens[i_n + 1].item<int64_t>();
    const int64_t T_len = eos - bos;
    const int64_t NT = (T_len + BT - 1) / BT;

    for (int64_t i_h = 0; i_h < H; ++i_h) {
      torch::Tensor h_state;
      if (h0_fp32.defined()) {
        h_state = h0_fp32[i_n][i_h].clone();
      } else {
        h_state = torch::zeros({K, V}, torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
      }
      const int64_t k_head = i_h / (H / Hg);

      for (int64_t i_t = 0; i_t < NT; ++i_t) {
        const int64_t t_start = i_t * BT;
        const int64_t t_end = std::min((i_t + 1) * BT, T_len);

        h[chunk_offset + i_t][i_h] = h_state;
        torch::Tensor k_chunk = k_fp32.slice(0, bos + t_start, bos + t_end).slice(1, k_head, k_head + 1).squeeze(1);
        torch::Tensor w_chunk = w_fp32.slice(0, bos + t_start, bos + t_end).slice(1, i_h, i_h + 1).squeeze(1);
        torch::Tensor v_chunk = u_fp32.slice(0, bos + t_start, bos + t_end).slice(1, i_h, i_h + 1).squeeze(1);

        torch::Tensor v_n = v_chunk - w_chunk.matmul(h_state);
        v_new.slice(0, bos + t_start, bos + t_end).slice(1, i_h, i_h + 1).squeeze(1) = v_n;

        if (g_fp32.defined()) {
          torch::Tensor g_chunk = g_fp32.slice(0, bos + t_start, bos + t_end).slice(1, i_h, i_h + 1).squeeze(1);
          const float g_last = g_chunk[-1].item<float>();
          v_n = v_n * torch::exp(g_last - g_chunk).unsqueeze(1);
          h_state = h_state * torch::exp(torch::tensor(g_last, torch::TensorOptions().dtype(torch::kFloat32).device(k.device())));
        }

        h_state = h_state + k_chunk.transpose(0, 1).matmul(v_n);
      }

      if (output_final_state) {
        final_state[i_n][i_h] = h_state;
      }
    }
    chunk_offset += NT;
  }

  return std::make_tuple(
      h.to(torch::kFloat16).unsqueeze(0),
      v_new.to(torch::kFloat16).unsqueeze(0),
      final_state.defined() ? final_state.to(torch::kFloat16) : torch::Tensor());
}

void run_chunk_gated_delta_rule_fwd_h_case(const ChunkGatedDeltaRuleFwdHTestCase& test_case) {
  ASSERT_GT(test_case.seqlens.size(), 0);
  ASSERT_GT(test_case.H, 0);
  ASSERT_GT(test_case.Hg, 0);
  ASSERT_GT(test_case.K, 0);
  ASSERT_GT(test_case.V, 0);
  ASSERT_GT(test_case.chunk_size, 0);
  ASSERT_LE(test_case.Hg, test_case.H);

  const auto npu_device = torch::Device("npu:0");
  const auto fp16_opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(npu_device);
  const auto fp32_opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(npu_device);

  torch::manual_seed(test_case.seed);

  const int64_t T_total = 0;
  for (const auto& seqlen : test_case.seqlens) {
    T_total += seqlen;
  }
  const int64_t N = test_case.seqlens.size();

  torch::Tensor cu_seqlens = torch::zeros({N + 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  for (int64_t i = 0; i < N; ++i) {
    cu_seqlens[i + 1] = cu_seqlens[i].item<int64_t>() + test_case.seqlens[i];
  }
  cu_seqlens = cu_seqlens.to(npu_device);

  torch::Tensor k = torch::randn({1, T_total, test_case.Hg, test_case.K}, fp16_opts) * 0.01;
  torch::Tensor w = torch::randn({1, T_total, test_case.H, test_case.K}, fp16_opts) * 0.01;
  torch::Tensor u = torch::randn({1, T_total, test_case.H, test_case.V}, fp16_opts) * 0.01;
  torch::Tensor g;
  if (test_case.use_g) {
    g = torch::randn({1, T_total, test_case.H}, fp32_opts) * -1.0;
  }
  torch::Tensor initial_state;
  if (test_case.use_initial_state) {
    initial_state = torch::randn({1, N, test_case.H, test_case.K, test_case.V}, fp16_opts) * 0.01;
  }

  auto [ref_h, ref_v_new, ref_ht] = torch_ref_chunk_gated_delta_rule(
      k, w, u, g, initial_state, true, test_case.chunk_size, cu_seqlens);

  auto [tilelang_h, tilelang_v_new, tilelang_ht] = chunk_gated_delta_rule_fwd_h(
      k, w, u, g, initial_state, true, test_case.chunk_size, true, cu_seqlens);

  torch::npu.synchronize();

  EXPECT_TRUE(torch::allclose(tilelang_h, ref_h, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "h mismatch: tilelang output differs from reference, case=" << test_case.name;
  EXPECT_TRUE(torch::allclose(tilelang_v_new, ref_v_new, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "v_new mismatch: tilelang output differs from reference, case=" << test_case.name;
  EXPECT_TRUE(torch::allclose(tilelang_ht, ref_ht, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "ht mismatch: tilelang output differs from reference, case=" << test_case.name;

  std::cout << "[chunk_gated_delta_rule_fwd_h_wrapper_test] case=" << test_case.name
            << " passed" << std::endl;
}

TEST_F(TileLangChunkGatedDeltaRuleFwdHWrapperTest, BasicSingleSequence) {
  const std::vector<ChunkGatedDeltaRuleFwdHTestCase> cases = {
      {.name = "single_2048_h8_hg4_k128_v128_bt64_useg",
       .seqlens = {2048},
       .H = 8,
       .Hg = 4,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .use_g = true,
       .use_initial_state = true,
       .seed = 20260507},
      {.name = "single_1024_h8_hg4_k128_v128_bt64_no_g",
       .seqlens = {1024},
       .H = 8,
       .Hg = 4,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .use_g = false,
       .use_initial_state = true,
       .seed = 20260508},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name);
    run_chunk_gated_delta_rule_fwd_h_case(test_case);
  }
}

TEST_F(TileLangChunkGatedDeltaRuleFwdHWrapperTest, MultiSequenceVarlen) {
  const std::vector<ChunkGatedDeltaRuleFwdHTestCase> cases = {
      {.name = "multi_512_512_h8_hg4_k128_v128_bt64",
       .seqlens = {512, 512},
       .H = 8,
       .Hg = 4,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .use_g = true,
       .use_initial_state = true,
       .seed = 20260509},
      {.name = "multi_128_256_512_h8_hg4_k128_v128_bt64",
       .seqlens = {128, 256, 512},
       .H = 8,
       .Hg = 4,
       .K = 128,
       .V = 128,
       .chunk_size = 64,
       .use_g = true,
       .use_initial_state = true,
       .seed = 20260510},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name);
    run_chunk_gated_delta_rule_fwd_h_case(test_case);
  }
}

TEST_F(TileLangChunkGatedDeltaRuleFwdHWrapperTest, SecondarySpecialization) {
  const std::vector<ChunkGatedDeltaRuleFwdHTestCase> cases = {
      {.name = "secondary_h4_hg2_k64_v64_bt32",
       .seqlens = {512},
       .H = 4,
       .Hg = 2,
       .K = 64,
       .V = 64,
       .chunk_size = 32,
       .use_g = true,
       .use_initial_state = true,
       .seed = 20260511},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name);
    run_chunk_gated_delta_rule_fwd_h_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang