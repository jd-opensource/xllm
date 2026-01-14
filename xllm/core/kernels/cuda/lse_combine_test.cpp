/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/kernels/cuda/cuda_ops_api.h"
#include "core/kernels/cuda/function_factory.h"

namespace xllm::kernel::cuda {

// NOTE: This test intentionally provides a local implementation of
// `xllm::kernel::cuda::batch_decode` to avoid linking the production
// implementation in `batch_decode.cpp`, which depends on NVTX (`NvtxRange`).
// With this definition present, the linker won't pull `batch_decode.cpp.o`
// out of `libcuda_kernels.a`, so the test has no NVTX dependency.
void batch_decode_test(torch::Tensor float_workspace_buffer,
                       torch::Tensor int_workspace_buffer,
                       torch::Tensor page_locked_int_workspace_buffer,
                       torch::Tensor query,
                       torch::Tensor k_cache,
                       torch::Tensor v_cache,
                       torch::Tensor paged_kv_indptr,
                       torch::Tensor paged_kv_indices,
                       torch::Tensor paged_kv_last_page_len,
                       int64_t window_left,
                       double sm_scale,
                       torch::Tensor output,
                       std::optional<torch::Tensor>& output_lse,
                       bool enable_cuda_graph,
                       std::optional<torch::Tensor>& plan_info) {
  std::string uri = get_batch_decode_uri(query.scalar_type(),
                                         k_cache.scalar_type(),
                                         output.scalar_type(),
                                         paged_kv_indptr.scalar_type(),
                                         query.size(-1),
                                         v_cache.size(-1),
                                         /*pos_encoding_mode=*/0,
                                         /*use_sliding_window=*/false,
                                         /*use_logits_soft_cap=*/false);

  torch::Tensor paged_kv_indptr_host = paged_kv_indptr.to(torch::kCPU);
  int64_t batch_size = paged_kv_last_page_len.size(0);

  torch::Tensor empty_q_data =
      torch::empty({0}, torch::TensorOptions().dtype(query.scalar_type()));
  torch::Tensor empty_kv_data =
      torch::empty({0}, torch::TensorOptions().dtype(k_cache.scalar_type()));

  torch::Tensor plan_info_tensor;
  if (plan_info.has_value()) {
    plan_info_tensor = *plan_info;
  } else {
    plan_info_tensor =
        FunctionFactory::get_instance().decode_plan_func(uri).call(
            float_workspace_buffer,
            int_workspace_buffer,
            page_locked_int_workspace_buffer,
            paged_kv_indptr_host,
            batch_size,
            query.size(1),    // num_qo_heads
            k_cache.size(2),  // num_kv_heads
            k_cache.size(1),  // block_size
            enable_cuda_graph,
            window_left,
            /*logits_soft_cap=*/0.0,
            query.size(-1),    // head_dim_qk
            v_cache.size(-1),  // head_dim_vo
            empty_q_data,
            empty_kv_data);
    // Cache plan for repeated calls (e.g. iterative stress tests).
    plan_info = plan_info_tensor;
  }

  FunctionFactory::get_instance().decode_run_func(uri).call(
      float_workspace_buffer,
      int_workspace_buffer,
      plan_info_tensor,
      query,
      k_cache,
      v_cache,
      paged_kv_indptr,
      paged_kv_indices,
      paged_kv_last_page_len,
      output,
      output_lse,
      /*kv_layout_code=*/0,  // NHD layout
      window_left,
      support_pdl(),
      /*maybe_alibi_slopes=*/std::optional<torch::Tensor>(),
      /*logits_soft_cap=*/0.0,
      sm_scale,
      /*rope_rcp_scale=*/1.0,
      /*rope_rcp_theta=*/1.0 / 10000.0);
}

namespace test {
namespace {

inline void log_tensor_stats(const char* name,
                             const torch::Tensor& t,
                             const torch::TensorOptions& float_options) {
  // Reduce in FP32 for stable stats.
  torch::Tensor tf = t.to(float_options);
  const double tmin = tf.min().item<double>();
  const double tmax = tf.max().item<double>();
  const double tmean = tf.mean().item<double>();
  // std() default is unbiased; for debug, biased is fine and faster.
  const double tstd = tf.std(/*unbiased=*/false).item<double>();
  const double tabsmax = tf.abs().max().item<double>();
}

// Shape a tensor's value range to mimic real-model ranges:
// - apply a base scale to widen typical values
// - apply a sparse outlier multiplier to create rare large-magnitude values
// This keeps determinism by setting a local seed per call.
inline void apply_scale_with_sparse_outliers(
    torch::Tensor& t,
    double base_scale,
    double outlier_prob,
    double outlier_mult,
    const torch::TensorOptions& float_options,
    uint64_t seed) {
  TORCH_CHECK(outlier_prob >= 0.0 && outlier_prob <= 1.0,
              "outlier_prob must be in [0,1]");
  TORCH_CHECK(outlier_mult >= 1.0, "outlier_mult must be >= 1");
  torch::manual_seed(seed);
  auto tf = t.to(float_options);
  tf.mul_(base_scale);
  if (outlier_prob > 0.0 && outlier_mult > 1.0) {
    auto mask = torch::rand(tf.sizes(), float_options).lt(outlier_prob);
    // Multiply a sparse subset by outlier_mult.
    auto factor = torch::ones(tf.sizes(), float_options);
    factor.masked_fill_(mask, outlier_mult);
    tf.mul_(factor);
  }
  t.copy_(tf.to(t.options()));
}

struct PagedKV {
  // Cache layout: [num_pages, block_size(=1), num_kv_heads, head_dim]
  torch::Tensor k_cache;
  torch::Tensor v_cache;
  torch::Tensor indptr;         // [B+1] int32, on CUDA
  torch::Tensor indices;        // [nnz_pages] int32, on CUDA
  torch::Tensor last_page_len;  // [B] int32, on CUDA

  // Host-side copies to help building composed layouts.
  std::vector<int32_t> indptr_h;
  std::vector<int32_t> indices_h;
};

PagedKV make_block1_paged_kv(const torch::Device& device,
                             torch::ScalarType kv_dtype,
                             int64_t num_kv_heads,
                             int64_t head_dim,
                             const std::vector<int32_t>& lens_per_batch,
                             int64_t total_pages,
                             int64_t max_decode_step,
                             int64_t beam_size) {
  TORCH_CHECK(device.is_cuda(), "make_block1_paged_kv expects CUDA device");
  const int64_t B = static_cast<int64_t>(lens_per_batch.size());

  std::vector<int32_t> indptr_h;
  int32_t sum_lens = 0;
  if (max_decode_step > 0) {
    indptr_h.reserve(B + 1);
    indptr_h.push_back(0);
    // NOTE: total pages is more than the sum of lens_per_batch
    for (int32_t len : lens_per_batch) {
      TORCH_CHECK(len > 0, "lens_per_batch must all be > 0");
      sum_lens += len;
      indptr_h.push_back(sum_lens);
    }
  } else {
    indptr_h.reserve(B * beam_size + 1);
    indptr_h.push_back(0);
    for (int beam_id = 0; beam_id < beam_size; beam_id++) {
      for (int32_t len : lens_per_batch) {
        sum_lens += len;
        indptr_h.push_back(sum_lens);
      }
    }
  }

  // Indices are simply 0..total_pages-1 (grouped by batch using indptr).
  std::vector<int32_t> indices_h;

  if (max_decode_step > 0) {
    indices_h.reserve(sum_lens);
    for (int batch_id = 0; batch_id < B; batch_id++) {
      int32_t len = lens_per_batch[batch_id];
      for (int32_t i = 0; i < len; ++i) {
        indices_h.push_back(batch_id * max_decode_step + i);
      }
    }
  } else {
    sum_lens = sum_lens / beam_size;
    indices_h.reserve(sum_lens * beam_size);
    for (int beam_id = 0; beam_id < beam_size; beam_id++) {
      for (int32_t i = 0; i < sum_lens; ++i) {
        indices_h.push_back(i);
      }
    }
  }

  auto kv_options = torch::TensorOptions().device(device).dtype(kv_dtype);
  torch::Tensor k_cache =
      torch::randn({total_pages, 1, num_kv_heads, head_dim}, kv_options);
  torch::Tensor v_cache =
      torch::randn({total_pages, 1, num_kv_heads, head_dim}, kv_options);

  auto idx_options = torch::TensorOptions().device(device).dtype(torch::kInt32);
  torch::Tensor indptr =
      torch::from_blob(indptr_h.data(),
                       {static_cast<int64_t>(indptr_h.size())},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone()
          .to(device);
  torch::Tensor indices =
      torch::from_blob(indices_h.data(),
                       {static_cast<int64_t>(indices_h.size())},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone()
          .to(device);
  // `paged_kv_last_page_len` length must match the effective batch size used by
  // FlashInfer decode plan/kernel.
  // - max_decode_step > 0  : `lens_per_batch` already enumerates sequences, so
  //                          batch_size == B
  // - max_decode_step <= 0 : we expand `indptr_h` for beam search, so
  //                          batch_size == B * beam_size
  const int64_t effective_B = (max_decode_step > 0) ? B : (B * beam_size);
  torch::Tensor last_page_len =
      torch::full({effective_B}, 1, idx_options);  // block_size=1 => always 1

  PagedKV out;
  out.k_cache = k_cache;
  out.v_cache = v_cache;
  out.indptr = indptr;
  out.indices = indices;
  out.last_page_len = last_page_len;
  out.indptr_h = std::move(indptr_h);
  out.indices_h = std::move(indices_h);
  return out;
}

std::vector<int32_t> build_full_indices_host(const PagedKV& shared,
                                             const PagedKV& unshared,
                                             int32_t shared_page_offset) {
  TORCH_CHECK(shared.indptr_h.size() == unshared.indptr_h.size(),
              "shared/unshared batch size mismatch");
  const int64_t B = static_cast<int64_t>(shared.indptr_h.size()) - 1;

  std::vector<int32_t> full_indices;
  full_indices.reserve(shared.indices_h.size() + unshared.indices_h.size());

  for (int64_t b = 0; b < B; ++b) {
    const int32_t s0 = shared.indptr_h[b];
    const int32_t s1 = shared.indptr_h[b + 1];
    const int32_t u0 = unshared.indptr_h[b];
    const int32_t u1 = unshared.indptr_h[b + 1];

    // shared indices for this batch
    for (int32_t i = s0; i < s1; ++i) {
      full_indices.push_back(shared.indices_h[i]);
    }
    // unshared indices for this batch (offset into concatenated cache)
    for (int32_t i = u0; i < u1; ++i) {
      full_indices.push_back(unshared.indices_h[i] + shared_page_offset);
    }
  }

  return full_indices;
}

}  // namespace

class FlashInferLseCombineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    device_ = torch::Device(torch::kCUDA);
  }

  torch::Device device_ = torch::kCPU;
};

void prinf_paged_kv(const PagedKV& paged_kv) {
  LOG(INFO) << "k_cache shape: " << paged_kv.k_cache.sizes();
  LOG(INFO) << "v_cache shape: " << paged_kv.v_cache.sizes();
  LOG(INFO) << "indptr shape: " << paged_kv.indptr.sizes();
  LOG(INFO) << "indices shape: " << paged_kv.indices.sizes();
  LOG(INFO) << "last_page_len shape: " << paged_kv.last_page_len.sizes();
  LOG(INFO) << " cout tensor values:";
  LOG(INFO) << "indptr: " << paged_kv.indptr;
  LOG(INFO) << "indices: " << paged_kv.indices;
  LOG(INFO) << "last_page_len: " << paged_kv.last_page_len;
  // LOG(INFO) << "k_cache: " << paged_kv.k_cache;
  // LOG(INFO) << "v_cache: " << paged_kv.v_cache;
}

TEST_F(FlashInferLseCombineTest, TwoDecodePlusLseCombineMatchesSingleDecode) {
  // This unit test validates:
  //   decode(shared_kv) + decode(unshared_kv) + lse_combine
  // is numerically consistent with:
  //   decode(concat(shared_kv, unshared_kv))
  //
  // Preconditions:
  // - FlashInfer compiled ops must be discoverable via FLASHINFER_OPS_PATH.
  const char* ops_path_c = std::getenv("FLASHINFER_OPS_PATH");
  if (ops_path_c == nullptr || std::string(ops_path_c).empty()) {
    GTEST_SKIP()
        << "FLASHINFER_OPS_PATH not set; cannot load flashinfer decode op .so.";
  }

  // Use a common supported configuration.
  const int64_t B = 1;
  const int64_t num_heads = 16;
  const int64_t num_kv_heads = 8;
  const int64_t head_dim = 128;
  const int64_t beam_size = 2;
  const int64_t max_decode_step = 3;
  const int64_t total_pages_shared = 1024;
  const int64_t total_pages_unshared = 10 * beam_size * max_decode_step;
  const auto q_dtype = torch::kBFloat16;
  const auto kv_dtype = torch::kBFloat16;
  const auto o_dtype = torch::kBFloat16;
  const auto idx_dtype = torch::kInt32;

  // Workspace buffers (same sizing as FlashinferWorkspace).
  auto ws_cuda_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device_);
  auto float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device_);
  auto bf16_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device_);
  constexpr int64_t kFlashinferWorkspaceBufferSizeBytes = 128LL * 1024 * 1024;
  torch::Tensor float_ws =
      torch::empty({kFlashinferWorkspaceBufferSizeBytes}, ws_cuda_u8);
  torch::Tensor int_ws = torch::empty({8 * 1024 * 1024}, ws_cuda_u8);
  torch::Tensor page_locked_int_ws = torch::empty({int_ws.size(0)},
                                                  torch::TensorOptions()
                                                      .dtype(torch::kUInt8)
                                                      .device(torch::kCPU)
                                                      .pinned_memory(true));

  // Construct a minimal block_size=1 paged KV layout with ragged lengths.
  // Each "page" is a single token (block_size=1), making indices easy to build.
  const std::vector<int32_t> shared_lens = {5};
  const std::vector<int32_t> unshared_lens = {1, 1};
  std::vector<int32_t> shared_lens_repeated(shared_lens.size() * beam_size);
  for (int i = 0; i < shared_lens.size(); ++i) {
    for (int j = 0; j < beam_size; ++j) {
      shared_lens_repeated[i * beam_size + j] = shared_lens[i];
    }
  }
  PagedKV shared = make_block1_paged_kv(device_,
                                        kv_dtype,
                                        num_kv_heads,
                                        head_dim,
                                        shared_lens,
                                        total_pages_shared,
                                        0,
                                        beam_size);

  PagedKV unshared = make_block1_paged_kv(device_,
                                          kv_dtype,
                                          num_kv_heads,
                                          head_dim,
                                          unshared_lens,
                                          total_pages_unshared,
                                          max_decode_step,
                                          beam_size);
  // Full cache is concatenation along the page dimension.
  torch::Tensor k_full = torch::cat({shared.k_cache, unshared.k_cache}, 0);
  torch::Tensor v_full = torch::cat({shared.v_cache, unshared.v_cache}, 0);

  std::vector<int32_t> full_indptr_h;
  full_indptr_h.reserve(B * beam_size + 1);
  full_indptr_h.push_back(0);
  // repeat the shared_lens for beam_size times

  for (int64_t b = 0; b < B * beam_size; ++b) {
    const int32_t len_b = shared_lens_repeated[static_cast<size_t>(b)] +
                          unshared_lens[static_cast<size_t>(b)];
    full_indptr_h.push_back(full_indptr_h.back() + len_b);
  }
  std::vector<int32_t> full_indices_h = build_full_indices_host(
      shared,
      unshared,
      /*shared_page_offset=*/static_cast<int32_t>(shared.k_cache.size(0)));

  auto idx_options =
      torch::TensorOptions().device(device_).dtype(torch::kInt32);
  torch::Tensor full_indptr =
      torch::from_blob(full_indptr_h.data(),
                       {static_cast<int64_t>(full_indptr_h.size())},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone()
          .to(device_);
  torch::Tensor full_indices =
      torch::from_blob(full_indices_h.data(),
                       {static_cast<int64_t>(full_indices_h.size())},
                       torch::TensorOptions().dtype(torch::kInt32))
          .clone()
          .to(device_);
  torch::Tensor full_last_page_len =
      torch::full({B * beam_size}, 1, idx_options);  // block_size=1 => always 1
  // Random query for a single-step decode: [B, H, D].
  torch::manual_seed(0);
  auto q_options = torch::TensorOptions().device(device_).dtype(q_dtype);
  torch::Tensor query =
      torch::randn({B * beam_size, num_heads, head_dim}, q_options);

  const double sm_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const int64_t window_left = 0;
  const bool enable_cuda_graph = false;

  // ---------------------------------------------------------------------------
  // Mimic real-model numeric ranges (based on observed logs):
  // - query can have occasional large values (tens)
  // - key/value can have rare extreme outliers (hundreds)
  // We model this as base_scale + sparse outlier multiplier.
  // ---------------------------------------------------------------------------
  apply_scale_with_sparse_outliers(query,
                                   /*base_scale=*/5.0,
                                   /*outlier_prob=*/1e-3,
                                   /*outlier_mult=*/8.0,
                                   float_options,
                                   /*seed=*/100);
  apply_scale_with_sparse_outliers(shared.k_cache,
                                   /*base_scale=*/4.0,
                                   /*outlier_prob=*/1e-4,
                                   /*outlier_mult=*/40.0,
                                   float_options,
                                   /*seed=*/200);
  apply_scale_with_sparse_outliers(shared.v_cache,
                                   /*base_scale=*/4.0,
                                   /*outlier_prob=*/1e-4,
                                   /*outlier_mult=*/40.0,
                                   float_options,
                                   /*seed=*/201);
  apply_scale_with_sparse_outliers(unshared.k_cache,
                                   /*base_scale=*/4.0,
                                   /*outlier_prob=*/1e-4,
                                   /*outlier_mult=*/40.0,
                                   float_options,
                                   /*seed=*/300);
  apply_scale_with_sparse_outliers(unshared.v_cache,
                                   /*base_scale=*/4.0,
                                   /*outlier_prob=*/1e-4,
                                   /*outlier_mult=*/40.0,
                                   float_options,
                                   /*seed=*/301);
  // Rebuild concatenated caches after scaling.
  k_full = torch::cat({shared.k_cache, unshared.k_cache}, 0);
  v_full = torch::cat({shared.v_cache, unshared.v_cache}, 0);

  // log_tensor_stats("query(init)", query, float_options);
  // log_tensor_stats("shared.k_cache", shared.k_cache, float_options);
  // log_tensor_stats("shared.v_cache", shared.v_cache, float_options);
  // log_tensor_stats("unshared.k_cache", unshared.k_cache, float_options);
  // log_tensor_stats("unshared.v_cache", unshared.v_cache, float_options);
  // log_tensor_stats("k_full", k_full, float_options);
  // log_tensor_stats("v_full", v_full, float_options);

  // Path A: two decodes + lse_combine.
  // Initialize outputs to detect if kernel fails to write them
  torch::Tensor o_shared = torch::zeros_like(query).to(float_options);
  torch::Tensor o_unshared = torch::zeros_like(query).to(float_options);
  torch::Tensor lse_shared =
      torch::zeros({B * beam_size, num_heads, 1}, float_options);
  torch::Tensor lse_unshared =
      torch::zeros({B * beam_size, num_heads, 1}, float_options);

  std::optional<torch::Tensor> lse_shared_opt = lse_shared;
  std::optional<torch::Tensor> lse_unshared_opt = lse_unshared;
  std::optional<torch::Tensor> plan_shared = std::nullopt;
  std::optional<torch::Tensor> plan_unshared = std::nullopt;

  batch_decode_test(float_ws,
                    int_ws,
                    page_locked_int_ws,
                    query,
                    shared.k_cache,
                    shared.v_cache,
                    shared.indptr,
                    shared.indices,
                    shared.last_page_len,
                    window_left,
                    sm_scale,
                    o_shared,
                    lse_shared_opt,
                    enable_cuda_graph,
                    plan_shared);

  batch_decode_test(float_ws,
                    int_ws,
                    page_locked_int_ws,
                    query,
                    unshared.k_cache,
                    unshared.v_cache,
                    unshared.indptr,
                    unshared.indices,
                    unshared.last_page_len,
                    window_left,
                    sm_scale,
                    o_unshared,
                    lse_unshared_opt,
                    enable_cuda_graph,
                    plan_unshared);

  torch::Tensor o_combined = torch::zeros_like(query).to(float_options);
  lse_combine(o_combined, o_shared, lse_shared, o_unshared, lse_unshared);

  // Path B: single decode over concatenated KV.
  torch::Tensor o_full = torch::zeros_like(query).to(float_options);
  torch::Tensor lse_full =
      torch::zeros({B * beam_size, num_heads, 1}, float_options);
  std::optional<torch::Tensor> lse_full_opt = lse_full;
  std::optional<torch::Tensor> plan_full = std::nullopt;

  batch_decode_test(float_ws,
                    int_ws,
                    page_locked_int_ws,
                    query,
                    k_full,
                    v_full,
                    full_indptr,
                    full_indices,
                    full_last_page_len,
                    window_left,
                    sm_scale,
                    o_full,
                    lse_full_opt,
                    enable_cuda_graph,
                    plan_full);

  // Compare.
  if (torch::any(torch::isnan(o_full)).item<bool>()) {
    LOG(ERROR) << "o_full contains NaN!";
  }
  if (torch::any(torch::isnan(o_combined)).item<bool>()) {
    LOG(ERROR) << "o_combined contains NaN!";
  }
  if (torch::any(torch::isnan(o_shared)).item<bool>()) {
    LOG(ERROR) << "o_shared contains NaN!";
  }
  if (torch::any(torch::isnan(o_unshared)).item<bool>()) {
    LOG(ERROR) << "o_unshared contains NaN!";
  }
  if (torch::any(torch::isnan(lse_shared)).item<bool>()) {
    LOG(ERROR) << "lse_shared contains NaN!";
  }
  if (torch::any(torch::isnan(lse_unshared)).item<bool>()) {
    LOG(ERROR) << "lse_unshared contains NaN!";
  }

  auto diff = (o_full.to(float_options) - o_combined.to(float_options)).abs();

  const double max_abs = diff.max().item<double>();
  // Compare in FP32 to avoid BF16 rounding inside the checker itself.
  // Tolerance should be BF16-scale (bf16 mantissa ~= 1/128 ~ 0.0078).
  EXPECT_TRUE(torch::allclose(o_full.to(float_options),
                              o_combined.to(float_options),
                              /*rtol=*/1e-2,
                              /*atol=*/1e-2))
      << "max_abs_diff=" << max_abs;

  // ---------------------------------------------------------------------------
  // Iterative "model-like" stress test:
  // Treat:
  // - Path A output (o_combined) as the next-step query for (shared+unshared),
  // - Path B output (o_full) as the next-step query for (full),
  // and repeat multiple times to observe potential error accumulation.
  //
  // Note: This does NOT model KV-cache growth; it isolates numerical divergence
  // caused by different compute/accumulation paths under fixed KV.
  // ---------------------------------------------------------------------------
  const int64_t num_iters = 32;
  torch::Tensor query_a = query.clone();
  torch::Tensor query_b = query.clone();
  double max_diff_overall = 0.0;
  double max_diff_last = 0.0;

  // Reuse plans across iterations (shape/dtype/layout are constant).
  std::optional<torch::Tensor> plan_shared_it = std::nullopt;
  std::optional<torch::Tensor> plan_unshared_it = std::nullopt;
  std::optional<torch::Tensor> plan_full_it = std::nullopt;

  for (int64_t it = 0; it < num_iters; ++it) {
    // if (it == 0 || it == num_iters - 1 || it % 8 == 0) {
    //   log_tensor_stats("[iterative] query_a", query_a, float_options);
    //   log_tensor_stats("[iterative] query_b", query_b, float_options);
    // }
    // Path A.
    torch::Tensor o_shared_it = torch::zeros_like(query_a).to(float_options);
    torch::Tensor o_unshared_it = torch::zeros_like(query_a).to(float_options);
    torch::Tensor lse_shared_it =
        torch::zeros({B * beam_size, num_heads, 1}, float_options);
    torch::Tensor lse_unshared_it =
        torch::zeros({B * beam_size, num_heads, 1}, float_options);
    std::optional<torch::Tensor> lse_shared_it_opt = lse_shared_it;
    std::optional<torch::Tensor> lse_unshared_it_opt = lse_unshared_it;

    batch_decode_test(float_ws,
                      int_ws,
                      page_locked_int_ws,
                      query_a,
                      shared.k_cache,
                      shared.v_cache,
                      shared.indptr,
                      shared.indices,
                      shared.last_page_len,
                      window_left,
                      sm_scale,
                      o_shared_it,
                      lse_shared_it_opt,
                      enable_cuda_graph,
                      plan_shared_it);

    batch_decode_test(float_ws,
                      int_ws,
                      page_locked_int_ws,
                      query_a,
                      unshared.k_cache,
                      unshared.v_cache,
                      unshared.indptr,
                      unshared.indices,
                      unshared.last_page_len,
                      window_left,
                      sm_scale,
                      o_unshared_it,
                      lse_unshared_it_opt,
                      enable_cuda_graph,
                      plan_unshared_it);

    torch::Tensor o_combined_it = torch::zeros_like(query_a).to(float_options);
    lse_combine(o_combined_it,
                o_shared_it,
                lse_shared_it,
                o_unshared_it,
                lse_unshared_it);

    // Path B.
    torch::Tensor o_full_it = torch::zeros_like(query_b).to(float_options);
    torch::Tensor lse_full_it =
        torch::zeros({B * beam_size, num_heads, 1}, float_options);
    std::optional<torch::Tensor> lse_full_it_opt = lse_full_it;

    batch_decode_test(float_ws,
                      int_ws,
                      page_locked_int_ws,
                      query_b,
                      k_full,
                      v_full,
                      full_indptr,
                      full_indices,
                      full_last_page_len,
                      window_left,
                      sm_scale,
                      o_full_it,
                      lse_full_it_opt,
                      enable_cuda_graph,
                      plan_full_it);

    // Safety: no NaNs.
    ASSERT_FALSE(torch::any(torch::isnan(o_full_it)).item<bool>())
        << "NaN in o_full at iter " << it;
    ASSERT_FALSE(torch::any(torch::isnan(o_combined_it)).item<bool>())
        << "NaN in o_combined at iter " << it;

    // if (it == 0 || it == num_iters - 1 || it % 8 == 0) {
    //   log_tensor_stats(
    //       "[iterative] o_combined_it", o_combined_it, float_options);
    //   log_tensor_stats("[iterative] o_full_it", o_full_it, float_options);
    // }

    auto diff_it =
        (o_full_it.to(float_options) - o_combined_it.to(float_options)).abs();
    const double max_diff_it = diff_it.max().item<double>();
    max_diff_last = max_diff_it;
    max_diff_overall = std::max(max_diff_overall, max_diff_it);

    // Feed outputs back as the next-step queries (model-like repeated calls).
    query_a = o_combined_it.to(bf16_options);
    query_b = o_full_it.to(bf16_options);
  }
}

}  // namespace test
}  // namespace xllm::kernel::cuda
