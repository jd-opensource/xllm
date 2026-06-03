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
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int32_t kWidth = 4;
constexpr int32_t kDim = 2048;
constexpr int32_t kStateLen = kWidth - 1;

class TileLangCausalConv1dWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

torch::Tensor causal_conv1d_cpu_ref(const torch::Tensor& x,
                                    const torch::Tensor& conv_state,
                                    const torch::Tensor& weight,
                                    const torch::Tensor& bias,
                                    const torch::Tensor& cu_seqlens,
                                    const torch::Tensor& init_indices,
                                    const torch::Tensor& current_indices,
                                    bool has_silu) {
  // Generate CPU reference matching kernel semantics.
  // x: [total_tokens, dim]
  // conv_state: [cache_lines, dim, state_len]  (PyTorch convention)
  // weight: [dim, width]                       (PyTorch convention)
  // bias: [dim]
  // cu_seqlens: [batch+1] int32
  // init_indices: [batch] int32
  // current_indices: [batch] int32
  const int64_t batch = cu_seqlens.size(0) - 1;
  const int64_t dim = x.size(1);
  const int64_t width = weight.size(1);
  const int64_t hist_len = width - 1;
  const int64_t total_tokens = x.size(0);

  auto x_f = x.to(torch::kFloat32);
  auto weight_f = weight.to(torch::kFloat32);
  auto conv_state_f = conv_state.to(torch::kFloat32).clone();
  auto bias_f = bias.to(torch::kFloat32);
  auto cu_seqlens_l = cu_seqlens.to(torch::kInt64);
  auto init_l = init_indices.to(torch::kInt64);
  auto current_l = current_indices.to(torch::kInt64);

  auto out = torch::zeros({total_tokens, dim}, torch::kFloat32);

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t read_line = init_l[b].item<int64_t>();
    const int64_t write_line = current_l[b].item<int64_t>();
    const int64_t qs = cu_seqlens_l[b].item<int64_t>();
    const int64_t qe = cu_seqlens_l[b + 1].item<int64_t>();

    // Load history from conv_state
    std::vector<torch::Tensor> history;
    for (int64_t h = 0; h < hist_len; ++h) {
      history.push_back(
          conv_state_f[read_line].index({torch::indexing::Slice(), h}).clone());
    }

    for (int64_t t = 0; t < (qe - qs); ++t) {
      auto acc = torch::zeros({dim}, torch::kFloat32);
      // Convolution with history
      for (int64_t w = 0; w < hist_len; ++w) {
        acc += weight_f.index({torch::indexing::Slice(), w}) * history[w];
      }
      // Current input
      acc +=
          weight_f.index({torch::indexing::Slice(), width - 1}) * x_f[qs + t];
      // Bias
      acc += bias_f;
      // Activation (silu)
      if (has_silu) {
        acc = acc / (1.0F + torch::exp(-acc));
      }
      out[qs + t] = acc;

      // Shift history
      for (int64_t h = 0; h < hist_len - 1; ++h) {
        history[h] = history[h + 1].clone();
      }
      history[hist_len - 1] = x_f[qs + t].clone();
    }

    // Write back last positions to conv_state
    for (int64_t h = 0; h < hist_len; ++h) {
      const int64_t idx = (qe - qs) - hist_len + h;
      if (idx >= 0) {
        conv_state_f[write_line].index_put_({torch::indexing::Slice(), h},
                                            x_f[qs + idx]);
      }
    }
  }

  conv_state.copy_(conv_state_f.to(conv_state.dtype()));
  return out.to(x.dtype());
}

torch::Tensor prepare_weight_t(const torch::Tensor& weight) {
  // weight [dim, width] -> weight_t [width, dim]
  return weight.transpose(0, 1).contiguous();
}

torch::Tensor prepare_conv_state_t(const torch::Tensor& conv_state) {
  // conv_state [cache_lines, dim, state_len] -> [cache_lines, state_len, dim]
  return conv_state.permute({0, 2, 1}).contiguous();
}

torch::Tensor conv_state_t_to_pytorch(const torch::Tensor& conv_state_t) {
  // conv_state_t [cache_lines, state_len, dim] -> [cache_lines, dim, state_len]
  return conv_state_t.permute({0, 2, 1}).contiguous();
}

struct CausalConv1dTestCase {
  std::string name;
  int64_t total_tokens;
  int64_t batch_size;
  int64_t seqlen;
  bool has_silu;
  int64_t num_cache_lines;
  int64_t seed;
};

void run_causal_conv1d_case(const CausalConv1dTestCase& test_case) {
  ASSERT_GT(test_case.total_tokens, 0);
  ASSERT_GT(test_case.batch_size, 0);
  ASSERT_GT(test_case.seqlen, 0);
  ASSERT_EQ(test_case.total_tokens, test_case.batch_size * test_case.seqlen);
  ASSERT_GT(test_case.num_cache_lines, 0);

  const auto npu_device = torch::Device("npu:0");
  const auto fp16_opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(npu_device);
  const auto i32_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(npu_device);

  torch::manual_seed(test_case.seed);

  // Create inputs in PyTorch-native layout
  auto x_raw = torch::randn({test_case.total_tokens, kDim}, fp16_opts);
  auto conv_state_raw =
      torch::randn({test_case.num_cache_lines, kDim, kStateLen}, fp16_opts);
  auto weight_raw = torch::randn({kDim, kWidth}, fp16_opts);
  auto bias_raw = torch::randn({kDim}, fp16_opts);

  // cu_seqlens: [0, seqlen, 2*seqlen, ...]
  auto cu_seqlens = torch::empty({test_case.batch_size + 1}, i32_opts);
  for (int64_t b = 0; b <= test_case.batch_size; ++b) {
    cu_seqlens[b] = static_cast<int32_t>(b * test_case.seqlen);
  }

  auto init_indices = torch::arange(0, test_case.batch_size, i32_opts);
  auto current_indices = torch::arange(0, test_case.batch_size, i32_opts);
  auto initial_state_mode = torch::ones(test_case.batch_size, i32_opts);

  // CPU reference
  auto x_cpu = x_raw.cpu();
  auto cs_cpu = conv_state_raw.cpu().clone();
  auto w_cpu = weight_raw.cpu();
  auto b_cpu = bias_raw.cpu();
  auto cu_cpu = cu_seqlens.cpu();
  auto init_cpu = init_indices.cpu();
  auto curr_cpu = current_indices.cpu();

  auto golden = causal_conv1d_cpu_ref(x_cpu,
                                      cs_cpu,
                                      w_cpu,
                                      b_cpu,
                                      cu_cpu,
                                      init_cpu,
                                      curr_cpu,
                                      test_case.has_silu);

  // Prepare kernel-native layouts
  auto weight_t = prepare_weight_t(weight_raw);
  auto conv_state_t = prepare_conv_state_t(conv_state_raw);

  // Run wrapped kernel
  auto y = causal_conv1d(/*conv_state=*/conv_state_t,
                         /*x=*/x_raw,
                         /*weight=*/weight_t,
                         /*bias=*/bias_raw,
                         /*cu_seqlens=*/cu_seqlens,
                         /*init_indices=*/init_indices,
                         /*current_indices=*/current_indices,
                         /*initial_state_mode=*/initial_state_mode,
                         /*has_silu=*/test_case.has_silu);

  // Compare output
  auto max_diff = (y.cpu().to(torch::kFloat32) - golden.to(torch::kFloat32))
                      .abs()
                      .max()
                      .item<float>();
  std::cout << "[causal_conv1d_wrapper_test] case=" << test_case.name
            << ", max_diff=" << max_diff << std::endl;

  EXPECT_TRUE(torch::allclose(y.cpu().to(torch::kFloat32),
                              golden.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "causal_conv1d output differs from CPU reference, max_diff="
      << max_diff;

  // Compare conv_state
  auto cs_result = conv_state_t_to_pytorch(conv_state_t);
  auto cs_max_diff =
      (cs_result.cpu().to(torch::kFloat32) - cs_cpu.to(torch::kFloat32))
          .abs()
          .max()
          .item<float>();
  EXPECT_TRUE(torch::allclose(cs_result.cpu().to(torch::kFloat32),
                              cs_cpu.to(torch::kFloat32),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "conv_state mismatch, max_diff=" << cs_max_diff;
}

TEST_F(TileLangCausalConv1dWrapperTest, DecodeBatch1Silu) {
  const std::vector<CausalConv1dTestCase> cases = {
      {.name = "decode_bs1_sl1",
       .total_tokens = 1,
       .batch_size = 1,
       .seqlen = 1,
       .has_silu = true,
       .num_cache_lines = 4,
       .seed = 42},
  };
  for (const auto& tc : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << tc.name);
    run_causal_conv1d_case(tc);
  }
}

TEST_F(TileLangCausalConv1dWrapperTest, PrefillBatch2Silu) {
  const std::vector<CausalConv1dTestCase> cases = {
      {.name = "prefill_bs2_sl16",
       .total_tokens = 32,
       .batch_size = 2,
       .seqlen = 16,
       .has_silu = true,
       .num_cache_lines = 8,
       .seed = 20260301},
      {.name = "prefill_bs2_sl1",
       .total_tokens = 2,
       .batch_size = 2,
       .seqlen = 1,
       .has_silu = true,
       .num_cache_lines = 8,
       .seed = 42},
  };
  for (const auto& tc : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << tc.name);
    run_causal_conv1d_case(tc);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
