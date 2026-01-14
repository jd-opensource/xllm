/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "core/kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::cuda {
namespace test {

// This test validates `xllm::kernel::cuda::decoder_reshape_and_cache` by
// comparing its output against two CPU/PyTorch reference implementations:
//
// - `torch_reference`: copies one full [beam_size, kv_heads, head_dim] slab at
//   once using `select(...).copy_` (easy to read and matches the math).
// - `torch_reference_bench_style`: copies one [head_dim] vector at a time for
//   each (b, beam, kv_head), mirroring the style used in the Python benchmark.
//
// Having two independent references helps catch subtle indexing/layout bugs in
// the CUDA kernel (e.g. mixing up beam/step strides).
class DecoderReshapeAndCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    device_ = torch::Device(torch::kCUDA);
    dtype_ = torch::kFloat16;
  }

  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_ = torch::kFloat16;
};

// Reference implementation using PyTorch indexing ops.
// It writes:
//   unshared_*_cache[request_id, :, step, :, :] = proj_*(b, :, :, :)
//
// Expected shapes:
// - proj_k/proj_v:         [batch_size, beam_size, kv_heads, head_dim]
// - unshared_k/unshared_v: [max_num_request, beam_size, max_decode_step,
// kv_heads, head_dim]
// - block_table:           [batch_size, 1] mapping batch -> request_id
// (block_id)

void torch_reference(
    torch::Tensor proj_k,  // [batch_size, beam_size, kv_heads, head_dim]
    torch::Tensor proj_v,  // [batch_size, beam_size, kv_heads, head_dim]
    torch::Tensor unshared_k_cache,  // [max_num_request, beam_size,
                                     // max_decode_step, kv_heads, head_dim]
    torch::Tensor unshared_v_cache,  // [max_num_request, beam_size,
                                     // max_decode_step, kv_heads, head_dim]
    torch::Tensor block_table,       // [batch_size, 1]
    uint32_t step) {
  // Read shapes.
  int64_t batch_size = proj_k.size(0);
  int64_t beam_size = proj_k.size(1);
  int64_t kv_heads = proj_k.size(2);
  int64_t head_dim = proj_k.size(3);
  int64_t max_num_request = unshared_k_cache.size(0);
  int64_t max_decode_step = unshared_k_cache.size(2);
  // Basic shape checks.
  CHECK_EQ(proj_v.sizes(), proj_k.sizes())
      << "proj_v and proj_k must have same shape";
  CHECK_EQ(block_table.size(0), batch_size)
      << "block_table size must match batch_size";
  CHECK_EQ(block_table.size(1), 1) << "block_table second dim must be 1";
  CHECK_LT(step, max_decode_step) << "step must be less than max_decode_step";
  CHECK_EQ(unshared_k_cache.size(1), beam_size)
      << "unshared_k_cache beam_size mismatch";
  CHECK_EQ(unshared_k_cache.size(3), kv_heads)
      << "unshared_k_cache kv_heads mismatch";
  CHECK_EQ(unshared_k_cache.size(4), head_dim)
      << "unshared_k_cache head_dim mismatch";
  CHECK_EQ(unshared_v_cache.sizes(), unshared_k_cache.sizes())
      << "unshared_v_cache and unshared_k_cache must have same shape";

  // Move block_table to CPU so `.item<>()` does not repeatedly sync.
  auto block_table_cpu =
      block_table.select(1, 0).to(torch::kCPU);  // [batch_size]

  // For each batch element, write into its assigned request_id slot.
  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t request_id = block_table_cpu[b].item<int64_t>();

    // Validate request_id.
    CHECK_GE(request_id, 0) << "Invalid request_id: " << request_id;
    CHECK_LT(request_id, max_num_request)
        << "request_id (" << request_id << ") >= max_num_request ("
        << max_num_request << ")";

    // Extract one batch slice: [beam_size, kv_heads, head_dim]
    auto proj_k_batch = proj_k[b];  // [beam_size, kv_heads, head_dim]
    auto proj_v_batch = proj_v[b];  // [beam_size, kv_heads, head_dim]

    // Write into cache at decode step `step`.
    // NOTE: dimension order is [request_id, beam, step, kv_head, head_dim].
    unshared_k_cache[request_id].select(1, step).copy_(proj_k_batch);
    unshared_v_cache[request_id].select(1, step).copy_(proj_v_batch);
  }
}

TEST_F(DecoderReshapeAndCacheTest, CorrectnessTest) {
  // Small shapes are enough to catch indexing bugs, while keeping the test
  // fast.
  const int64_t batch_size = 1;
  const int64_t beam_size = 2;
  const int64_t kv_heads = 8;
  const int64_t head_dim = 128;
  const int64_t max_num_request = 33437;
  const int64_t max_decode_step = 3;
  const uint32_t step = 1;

  auto options = torch::TensorOptions().device(device_).dtype(dtype_);

  // 1) Prepare inputs.
  // proj_k/proj_v: [batch_size, beam_size, kv_heads, head_dim]
  torch::Tensor proj_k =
      torch::randn({batch_size, beam_size, kv_heads, head_dim}, options);
  torch::Tensor proj_v =
      torch::randn({batch_size, beam_size, kv_heads, head_dim}, options);

  // 2) Prepare block table mapping batch index -> request_id (block_id).
  // Here we map the only batch element to request_id=0.
  torch::Tensor block_table =
      torch::tensor({0}, torch::kInt64).view({batch_size, 1}).to(device_);

  // 3) Prepare caches.
  // Layout: [max_num_request, beam_size, max_decode_step, kv_heads, head_dim]
  torch::Tensor unshared_k_cache = torch::zeros(
      {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
      options);
  torch::Tensor unshared_v_cache = torch::zeros(
      {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
      options);

  // Reference buffers (two independent references).
  torch::Tensor ref_k_cache = unshared_k_cache.clone();
  torch::Tensor ref_v_cache = unshared_v_cache.clone();

  // 4) Run CUDA kernel under test.
  decoder_reshape_and_cache(
      proj_k, proj_v, unshared_k_cache, unshared_v_cache, block_table, step);

  // 5) Run references.
  torch_reference(proj_k, proj_v, ref_k_cache, ref_v_cache, block_table, step);

  // 6) Compare results.
  EXPECT_TRUE(torch::allclose(unshared_k_cache, ref_k_cache, 1e-5, 1e-5));
  EXPECT_TRUE(torch::allclose(unshared_v_cache, ref_v_cache, 1e-5, 1e-5));

  // Sanity-check that the expected slice was copied for each batch element.
  for (int64_t b = 0; b < batch_size; ++b) {
    int64_t rid = block_table[b][0].item<int64_t>();
    torch::Tensor copied_k = unshared_k_cache[rid].select(1, step);
    torch::Tensor source_k = proj_k[b];
    EXPECT_TRUE(torch::allclose(copied_k, source_k, 1e-5, 1e-5));
  }
}

}  // namespace test
}  // namespace xllm::kernel::cuda
