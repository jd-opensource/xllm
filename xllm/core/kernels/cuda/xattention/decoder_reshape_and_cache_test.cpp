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

#include "core/kernels/cuda/xattention/xattention_ops_api.h"

namespace xllm::kernel::cuda {
namespace test {

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

// This test validates `xllm::kernel::cuda::decoder_reshape_and_cache` against
// a PyTorch reference implementation that follows the current CUDA kernel
// contract.
//
// Kernel input contract:
// - proj_k/proj_v: [num_seqs, kv_heads, head_dim], where
//   num_seqs = batch_size * beam_size
// - block_table:   [num_seqs, 1], sequence-level block ids
//
// Cache layout:
// - unshared_k/v_cache: [max_num_request, beam_size, max_decode_step, kv_heads,
//   head_dim]
//
// Mapping used by both kernel and reference:
// - request_id = block_id / beam_size
// - beam_id    = block_id % beam_size
//
// The test checks:
// 1) full-tensor equivalence (CUDA output vs reference output)
// 2) per-sequence slice correctness at the target decode step
void torch_reference(const torch::Tensor& proj_k,
                     const torch::Tensor& proj_v,
                     torch::Tensor& unshared_k_cache,
                     torch::Tensor& unshared_v_cache,
                     const torch::Tensor& block_table,
                     const torch::Tensor& step) {
  const int64_t num_seqs = proj_k.size(0);
  const int64_t kv_heads = proj_k.size(1);
  const int64_t head_dim = proj_k.size(2);

  const int64_t max_num_request = unshared_k_cache.size(0);
  const int64_t beam_size = unshared_k_cache.size(1);
  const int64_t max_decode_step = unshared_k_cache.size(2);

  CHECK_EQ(proj_k.dim(), 3) << "proj_k must be 3-dimensional";
  CHECK_EQ(proj_v.dim(), 3) << "proj_v must be 3-dimensional";
  CHECK_EQ(proj_v.sizes(), proj_k.sizes())
      << "proj_v and proj_k must have same shape";

  CHECK_EQ(block_table.dim(), 2) << "block_table must be [num_seqs, 1]";
  CHECK_EQ(block_table.size(1), 1) << "block_table second dim must be 1";
  CHECK_EQ(block_table.size(0), num_seqs)
      << "block_table size must match num_seqs";

  CHECK_EQ(unshared_k_cache.dim(), 5)
      << "unshared_k_cache must be 5-dimensional";
  CHECK_EQ(unshared_v_cache.sizes(), unshared_k_cache.sizes())
      << "unshared_v_cache and unshared_k_cache must have same shape";
  CHECK_EQ(unshared_k_cache.size(3), kv_heads)
      << "unshared_k_cache kv_heads mismatch";
  CHECK_EQ(unshared_k_cache.size(4), head_dim)
      << "unshared_k_cache head_dim mismatch";

  CHECK_EQ(step.dim(), 1) << "step must be 1-dimensional";
  CHECK_EQ(step.size(0), 1) << "step must have shape [1]";
  const int64_t step_value = step[0].item<int64_t>();
  CHECK_GE(step_value, 0) << "step must be >= 0";
  CHECK_LT(step_value, max_decode_step)
      << "step must be less than max_decode_step";

  const int64_t max_num_seqs = max_num_request * beam_size;
  auto block_table_cpu = block_table.select(1, 0).to(torch::kCPU);

  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t block_id = block_table_cpu[seq_idx].item<int64_t>();
    CHECK_GE(block_id, 0) << "Invalid block_id: " << block_id;
    CHECK_LT(block_id, max_num_seqs)
        << "block_id (" << block_id << ") >= max_num_seqs (" << max_num_seqs
        << ")";

    const int64_t request_id = block_id / beam_size;
    const int64_t beam_id = block_id % beam_size;

    unshared_k_cache[request_id][beam_id]
        .select(0, step_value)
        .copy_(proj_k[seq_idx]);
    unshared_v_cache[request_id][beam_id]
        .select(0, step_value)
        .copy_(proj_v[seq_idx]);
  }
}

TEST_F(DecoderReshapeAndCacheTest, CorrectnessTest) {
  const int64_t batch_size = 1;
  const int64_t beam_size = 2;
  const int64_t num_seqs = batch_size * beam_size;
  const int64_t kv_heads = 8;
  const int64_t head_dim = 128;
  const int64_t max_num_request = 2;
  const int64_t max_decode_step = 3;

  torch::Tensor step = torch::tensor({1}, torch::kInt32).to(device_);

  const auto float_opts = torch::TensorOptions().device(device_).dtype(dtype_);
  const auto int_opts =
      torch::TensorOptions().device(device_).dtype(torch::kInt32);

  // Current kernel input shape: [num_seqs, kv_heads, head_dim]
  torch::Tensor proj_k =
      torch::randn({num_seqs, kv_heads, head_dim}, float_opts);
  torch::Tensor proj_v =
      torch::randn({num_seqs, kv_heads, head_dim}, float_opts);

  // Sequence-level block ids.
  torch::Tensor block_table =
      torch::arange(num_seqs, int_opts).view({num_seqs, 1});

  torch::Tensor unshared_k_cache = torch::zeros(
      {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
      float_opts);
  torch::Tensor unshared_v_cache = torch::zeros(
      {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
      float_opts);

  torch::Tensor ref_k_cache = unshared_k_cache.clone();
  torch::Tensor ref_v_cache = unshared_v_cache.clone();

  decoder_reshape_and_cache(
      proj_k, proj_v, unshared_k_cache, unshared_v_cache, block_table, step);

  torch_reference(proj_k, proj_v, ref_k_cache, ref_v_cache, block_table, step);

  EXPECT_TRUE(torch::allclose(unshared_k_cache, ref_k_cache, 1e-5, 1e-5));
  EXPECT_TRUE(torch::allclose(unshared_v_cache, ref_v_cache, 1e-5, 1e-5));

  const int64_t step_value = step[0].item<int64_t>();
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t block_id = block_table[seq_idx][0].item<int64_t>();
    const int64_t request_id = block_id / beam_size;
    const int64_t beam_id = block_id % beam_size;

    torch::Tensor copied_k =
        unshared_k_cache[request_id][beam_id].select(0, step_value);
    torch::Tensor source_k = proj_k[seq_idx];
    EXPECT_TRUE(torch::allclose(copied_k, source_k, 1e-5, 1e-5));
  }
}

}  // namespace test
}  // namespace xllm::kernel::cuda
