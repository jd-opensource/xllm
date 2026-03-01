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

#include "rope_wrapper.h"

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"

namespace xllm::kernel::npu::tilelang {
namespace {

#ifndef XLLM_TL_ROPE_HEAD_DIM
#define XLLM_TL_ROPE_HEAD_DIM 128
#endif

#ifndef XLLM_TL_ROPE_ROPE_DIM
#define XLLM_TL_ROPE_ROPE_DIM 128
#endif

class TileLangRopeWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct RopeTestCase {
  std::string name;
  int64_t num_tokens;
  int64_t num_heads;
  int64_t seed;
};

torch::Tensor torch_rope_ref(const torch::Tensor& x,
                             const torch::Tensor& sin,
                             const torch::Tensor& cos) {
  // x: [batch, head, dim]
  // cos/sin: [batch, dim] -> broadcast to [batch, head, dim]
  auto cos_ref = cos;
  auto sin_ref = sin;
  if (cos_ref.dim() == 2) {
    cos_ref = cos_ref.unsqueeze(1);
    sin_ref = sin_ref.unsqueeze(1);
  }

  auto x_fp32 = x.to(torch::kFloat32);
  auto cos_fp32 = cos_ref.to(torch::kFloat32);
  auto sin_fp32 = sin_ref.to(torch::kFloat32);

  auto x_reshaped =
      x_fp32.view({x_fp32.size(0), x_fp32.size(1), x_fp32.size(2) / 2, 2});
  auto x0 = x_reshaped.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              0});
  auto x1 = x_reshaped.index({torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              torch::indexing::Slice(),
                              1});
  auto x_rotated = torch::stack({-x1, x0}, /*dim=*/-1).flatten(-2);

  auto out = x_fp32 * cos_fp32 + x_rotated * sin_fp32;
  return out.to(torch::kBFloat16);
}

double measure_npu_event_ms(const std::function<void()>& fn,
                            int32_t device_id,
                            int warmup_iters = 5,
                            int measure_iters = 100) {
  CHECK_GT(measure_iters, 0) << "measure_iters must be > 0";
  CHECK_GE(warmup_iters, 0) << "warmup_iters must be >= 0";

  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  for (int i = 0; i < warmup_iters; ++i) {
    fn();
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS)
      << "warmup stream synchronize failed";

  aclrtEvent start_event = nullptr;
  aclrtEvent end_event = nullptr;
  CHECK_EQ(aclrtCreateEvent(&start_event), ACL_SUCCESS)
      << "aclrtCreateEvent(start) failed";
  CHECK_EQ(aclrtCreateEvent(&end_event), ACL_SUCCESS)
      << "aclrtCreateEvent(end) failed";

  CHECK_EQ(aclrtRecordEvent(start_event, stream), ACL_SUCCESS)
      << "aclrtRecordEvent(start) failed";
  for (int i = 0; i < measure_iters; ++i) {
    fn();
  }
  CHECK_EQ(aclrtRecordEvent(end_event, stream), ACL_SUCCESS)
      << "aclrtRecordEvent(end) failed";
  CHECK_EQ(aclrtSynchronizeEvent(end_event), ACL_SUCCESS)
      << "aclrtSynchronizeEvent(end) failed";

  float elapsed_ms = 0.0F;
  CHECK_EQ(aclrtEventElapsedTime(&elapsed_ms, start_event, end_event),
           ACL_SUCCESS)
      << "aclrtEventElapsedTime failed";
  CHECK_EQ(aclrtDestroyEvent(start_event), ACL_SUCCESS)
      << "aclrtDestroyEvent(start) failed";
  CHECK_EQ(aclrtDestroyEvent(end_event), ACL_SUCCESS)
      << "aclrtDestroyEvent(end) failed";

  return static_cast<double>(elapsed_ms) / static_cast<double>(measure_iters);
}

void run_apply_rotary_case(const RopeTestCase& test_case) {
  constexpr int64_t kInputDim = XLLM_TL_ROPE_ROPE_DIM;
  constexpr int64_t kRopeDim = XLLM_TL_ROPE_ROPE_DIM;

  ASSERT_GT(test_case.num_tokens, 0);
  ASSERT_GT(test_case.num_heads, 0);

  const auto npu_device = torch::Device("npu:0");
  const int32_t device_id = npu_device.index();
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);

  torch::manual_seed(test_case.seed);
  auto q_input = torch::randn(
      {test_case.num_tokens, test_case.num_heads, kInputDim}, bf16_opts);
  auto k_input = torch::randn(
      {test_case.num_tokens, test_case.num_heads, kInputDim}, bf16_opts);
  auto sin_cache = torch::randn({test_case.num_tokens, kRopeDim}, bf16_opts);
  auto cos_cache = torch::randn({test_case.num_tokens, kRopeDim}, bf16_opts);

  auto q_ref = torch_rope_ref(q_input, sin_cache, cos_cache);
  auto k_ref = torch_rope_ref(k_input, sin_cache, cos_cache);
  auto q = q_input.clone();
  auto k = k_input.clone();
  rope_in_place(q, sin_cache, cos_cache);
  rope_in_place(k, sin_cache, cos_cache);

  auto q_bench = q_input.clone();
  auto k_bench = k_input.clone();
  const double ref_elapsed_ms = measure_npu_event_ms(
      [&]() {
        [[maybe_unused]] auto q_ref_bench =
            torch_rope_ref(q_input, sin_cache, cos_cache);
        [[maybe_unused]] auto k_ref_bench =
            torch_rope_ref(k_input, sin_cache, cos_cache);
      },
      device_id);
  const double tl_elapsed_ms = measure_npu_event_ms(
      [&]() {
        rope_in_place(q_bench, sin_cache, cos_cache);
        rope_in_place(k_bench, sin_cache, cos_cache);
      },
      device_id);

  const double speedup =
      tl_elapsed_ms > 0.0 ? ref_elapsed_ms / tl_elapsed_ms : 0.0;
  std::cout << "[rope_wrapper_test] case=" << test_case.name
            << ", ref_ms=" << ref_elapsed_ms
            << ", tilelang_ms=" << tl_elapsed_ms << ", speedup=" << speedup
            << "x" << std::endl;

  auto q_max_diff = (q.to(torch::kFloat32) - q_ref.to(torch::kFloat32))
                        .abs()
                        .max()
                        .item<float>();
  auto k_max_diff = (k.to(torch::kFloat32) - k_ref.to(torch::kFloat32))
                        .abs()
                        .max()
                        .item<float>();

  EXPECT_TRUE(torch::allclose(q, q_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "q mismatch: tilelang output differs from interleaved rope reference"
      << ", max_diff=" << q_max_diff;
  EXPECT_TRUE(torch::allclose(k, k_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
      << "k mismatch: tilelang output differs from interleaved rope reference"
      << ", max_diff=" << k_max_diff;
}

TEST_F(TileLangRopeWrapperTest, ApplyRotaryMatchesNpuReference) {
  run_apply_rotary_case({
      .name = "baseline_16x4",
      .num_tokens = 16,
      .num_heads = 4,
      .seed = 20260213,
  });
}

TEST_F(TileLangRopeWrapperTest, ApplyRotaryMatchesNpuReferenceLargeTokens) {
  run_apply_rotary_case({
      .name = "large_tokens_2051x2",
      .num_tokens = 2051,
      .num_heads = 2,
      .seed = 20260213,
  });
}

TEST_F(TileLangRopeWrapperTest, ApplyRotaryMatchesNpuReferenceVariousShapes) {
  const std::vector<RopeTestCase> cases = {
      {.name = "tiny_1x1", .num_tokens = 1, .num_heads = 1, .seed = 101},
      {.name = "odd_tokens_7x3", .num_tokens = 7, .num_heads = 3, .seed = 102},
      {.name = "token_dim_64x4", .num_tokens = 64, .num_heads = 4, .seed = 107},
      {.name = "chunk_boundary_8x5",
       .num_tokens = 8,
       .num_heads = 5,
       .seed = 103},
      {.name = "cross_chunk_9x5", .num_tokens = 9, .num_heads = 5, .seed = 104},
      {.name = "head_dim_4x64", .num_tokens = 4, .num_heads = 64, .seed = 108},
      {.name = "medium_127x8", .num_tokens = 127, .num_heads = 8, .seed = 105},
      {.name = "large_heads_33x16",
       .num_tokens = 33,
       .num_heads = 16,
       .seed = 106},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name
                                      << ", num_tokens=" << test_case.num_tokens
                                      << ", num_heads=" << test_case.num_heads);
    run_apply_rotary_case(test_case);
  }
}

TEST_F(TileLangRopeWrapperTest,
       ApplyRotaryMatchesNpuReferenceLargeShape2048x576) {
#if XLLM_TL_ROPE_HEAD_DIM != 576 || XLLM_TL_ROPE_ROPE_DIM != 64
  GTEST_SKIP() << "requires compile dims head_dim=576, rope_dim=64";
#endif

  constexpr int64_t kNumHeads = 1;
  constexpr int64_t kFullHeadDim = 576;
  constexpr int64_t kStartDim = 512;
  constexpr int64_t kRopeDim = 64;

  const auto npu_device = torch::Device("npu:0");
  const int32_t device_id = npu_device.index();
  const auto bf16_opts =
      torch::TensorOptions().dtype(torch::kBFloat16).device(npu_device);

  const std::vector<RopeTestCase> cases = {
      {.name = "1x576_start512_rope64",
       .num_tokens = 1,
       .num_heads = kNumHeads,
       .seed = 20260226},
      {.name = "47x576_start512_rope64",
       .num_tokens = 47,
       .num_heads = kNumHeads,
       .seed = 20260301},
      {.name = "48x576_start512_rope64",
       .num_tokens = 48,
       .num_heads = kNumHeads,
       .seed = 20260302},
      {.name = "49x576_start512_rope64",
       .num_tokens = 49,
       .num_heads = kNumHeads,
       .seed = 20260303},
      {.name = "95x576_start512_rope64",
       .num_tokens = 95,
       .num_heads = kNumHeads,
       .seed = 20260304},
      {.name = "96x576_start512_rope64",
       .num_tokens = 96,
       .num_heads = kNumHeads,
       .seed = 20260305},
      {.name = "97x576_start512_rope64",
       .num_tokens = 97,
       .num_heads = kNumHeads,
       .seed = 20260306},
      {.name = "8x576_start512_rope64",
       .num_tokens = 8,
       .num_heads = kNumHeads,
       .seed = 20260227},
      {.name = "128x576_start512_rope64",
       .num_tokens = 128,
       .num_heads = kNumHeads,
       .seed = 20260228},
      {.name = "2048x576_start512_rope64",
       .num_tokens = 2048,
       .num_heads = kNumHeads,
       .seed = 20260225},
  };

  for (const auto& test_case : cases) {
    SCOPED_TRACE(::testing::Message() << "case=" << test_case.name
                                      << ", num_tokens=" << test_case.num_tokens
                                      << ", num_heads=" << test_case.num_heads);
    torch::manual_seed(test_case.seed);
    auto q_full = torch::randn(
        {test_case.num_tokens, test_case.num_heads, kFullHeadDim}, bf16_opts);
    auto k_full = torch::randn(
        {test_case.num_tokens, test_case.num_heads, kFullHeadDim}, bf16_opts);
    auto q = q_full.narrow(/*dim=*/2, /*start=*/kStartDim, /*length=*/kRopeDim);
    auto k = k_full.narrow(/*dim=*/2, /*start=*/kStartDim, /*length=*/kRopeDim);
    EXPECT_EQ(q.storage_offset(), kStartDim);
    EXPECT_EQ(k.storage_offset(), kStartDim);
    if (test_case.num_tokens * test_case.num_heads > 1) {
      EXPECT_FALSE(q.is_contiguous());
      EXPECT_FALSE(k.is_contiguous());
    }

    auto sin_cache = torch::randn({test_case.num_tokens, kRopeDim}, bf16_opts);
    auto cos_cache = torch::randn({test_case.num_tokens, kRopeDim}, bf16_opts);

    auto q_ref = torch_rope_ref(q, sin_cache, cos_cache);
    auto k_ref = torch_rope_ref(k, sin_cache, cos_cache);

    rope_in_place(q, sin_cache, cos_cache);
    rope_in_place(k, sin_cache, cos_cache);

    auto q_bench_full = q_full.clone();
    auto k_bench_full = k_full.clone();
    auto q_ref_bench_full = q_full.clone();
    auto k_ref_bench_full = k_full.clone();
    auto q_bench = q_bench_full.narrow(
        /*dim=*/2, /*start=*/kStartDim, /*length=*/kRopeDim);
    auto k_bench = k_bench_full.narrow(
        /*dim=*/2, /*start=*/kStartDim, /*length=*/kRopeDim);
    auto q_ref_bench = q_ref_bench_full.narrow(
        /*dim=*/2, /*start=*/kStartDim, /*length=*/kRopeDim);
    auto k_ref_bench = k_ref_bench_full.narrow(
        /*dim=*/2, /*start=*/kStartDim, /*length=*/kRopeDim);
    const double ref_elapsed_ms = measure_npu_event_ms(
        [&]() {
          [[maybe_unused]] auto q_ref_result =
              torch_rope_ref(q_ref_bench, sin_cache, cos_cache);
          [[maybe_unused]] auto k_ref_result =
              torch_rope_ref(k_ref_bench, sin_cache, cos_cache);
        },
        device_id);
    const double tl_elapsed_ms = measure_npu_event_ms(
        [&]() {
          rope_in_place(q_bench, sin_cache, cos_cache);
          rope_in_place(k_bench, sin_cache, cos_cache);
        },
        device_id);
    const double speedup =
        tl_elapsed_ms > 0.0 ? ref_elapsed_ms / tl_elapsed_ms : 0.0;
    std::cout << "[rope_wrapper_test] case=" << test_case.name
              << ", ref_ms=" << ref_elapsed_ms
              << ", tilelang_ms=" << tl_elapsed_ms << ", speedup=" << speedup
              << "x" << std::endl;

    auto q_max_diff = (q.to(torch::kFloat32) - q_ref.to(torch::kFloat32))
                          .abs()
                          .max()
                          .item<float>();
    auto k_max_diff = (k.to(torch::kFloat32) - k_ref.to(torch::kFloat32))
                          .abs()
                          .max()
                          .item<float>();
    EXPECT_TRUE(torch::allclose(q, q_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
        << "q mismatch for non-contiguous slice path, max_diff=" << q_max_diff;
    EXPECT_TRUE(torch::allclose(k, k_ref, /*rtol=*/1e-2, /*atol=*/1e-2))
        << "k mismatch for non-contiguous slice path, max_diff=" << k_max_diff;
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
