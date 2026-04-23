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
#include <torch_npu/torch_npu.h>

#include <string>
#include <vector>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TileLangModelInputBufferUpdaterWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

struct UpdaterTestCase {
  std::string name;
  int64_t actual_num_tokens;
  int64_t padded_num_tokens;
  int64_t actual_batch_size;
  int64_t actual_block_table_len;
  int64_t dst_block_table_stride;
  bool with_mrope;
  bool with_input_embedding;
  bool with_q_cu_seq_lens;
  bool with_linear_state_indices;
  c10::ScalarType embedding_dtype;
  int64_t hidden_size;
  int64_t seed;
};

struct UpdaterTestBuffers {
  ModelInputBufferUpdaterParams params;
  torch::Tensor ref_dst_tokens;
  torch::Tensor ref_dst_positions;
  torch::Tensor ref_dst_new_cache_slots;
  torch::Tensor ref_dst_q_seq_lens;
  torch::Tensor ref_dst_kv_seq_lens;
  torch::Tensor ref_dst_q_cu_seq_lens;
  torch::Tensor ref_dst_linear_state_indices;
  torch::Tensor ref_dst_block_tables;
  torch::Tensor ref_dst_input_embedding;
};

void expect_tensor_equal(const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         const std::string& name) {
  ASSERT_TRUE(torch::equal(actual.cpu(), expected.cpu()))
      << name << " mismatch\nactual=" << actual.cpu()
      << "\nexpected=" << expected.cpu();
}

UpdaterTestBuffers make_case(const UpdaterTestCase& test_case) {
  const torch::Device device("npu:0");
  const auto int_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  const auto emb_opts =
      torch::TensorOptions().dtype(test_case.embedding_dtype).device(device);

  torch::manual_seed(test_case.seed);

  const int64_t token_capacity =
      std::max<int64_t>(32, test_case.padded_num_tokens + 7);
  const int64_t batch_capacity =
      std::max<int64_t>(16, test_case.actual_batch_size + 5);

  UpdaterTestBuffers buffers;
  buffers.params.padded_num_tokens = test_case.padded_num_tokens;

  buffers.params.src_tokens =
      torch::randint(0, 32000, {test_case.actual_num_tokens}, int_opts);
  buffers.params.src_new_cache_slots =
      torch::randint(0, 4096, {test_case.actual_num_tokens}, int_opts);
  buffers.params.src_q_seq_lens =
      torch::randint(1, 3, {test_case.actual_batch_size}, int_opts);
  buffers.params.src_kv_seq_lens =
      torch::randint(1, 4096, {test_case.actual_batch_size}, int_opts);
  buffers.params.src_block_tables = torch::randint(
      0,
      4096,
      {test_case.actual_batch_size, test_case.actual_block_table_len},
      int_opts);

  if (test_case.with_mrope) {
    buffers.params.src_positions =
        torch::randint(0, 32000, {3, test_case.actual_num_tokens}, int_opts);
    buffers.params.dst_positions =
        torch::full({3, token_capacity}, -1, int_opts);
  } else {
    buffers.params.src_positions =
        torch::randint(0, 32000, {test_case.actual_num_tokens}, int_opts);
    buffers.params.dst_positions = torch::full({token_capacity}, -1, int_opts);
  }

  buffers.params.dst_tokens = torch::full({token_capacity}, -1, int_opts);
  buffers.params.dst_new_cache_slots =
      torch::full({token_capacity}, -1, int_opts);
  buffers.params.dst_q_seq_lens = torch::full({batch_capacity}, -1, int_opts);
  buffers.params.dst_kv_seq_lens = torch::full({batch_capacity}, -1, int_opts);
  buffers.params.dst_block_tables = torch::full(
      {batch_capacity, test_case.dst_block_table_stride}, -1, int_opts);

  if (test_case.with_q_cu_seq_lens) {
    buffers.params.src_q_cu_seq_lens =
        torch::randint(1, 4096, {test_case.actual_batch_size}, int_opts);
    buffers.params.dst_q_cu_seq_lens =
        torch::full({batch_capacity}, -1, int_opts);
  }

  if (test_case.with_linear_state_indices) {
    buffers.params.src_linear_state_indices =
        torch::randint(0, 4096, {test_case.actual_batch_size}, int_opts);
    buffers.params.dst_linear_state_indices =
        torch::full({batch_capacity}, -1, int_opts);
  }

  if (test_case.with_input_embedding) {
    buffers.params.src_input_embedding = torch::randn(
        {test_case.actual_num_tokens, test_case.hidden_size}, emb_opts);
    buffers.params.dst_input_embedding =
        torch::full({token_capacity, test_case.hidden_size}, -1, emb_opts);
  }

  buffers.ref_dst_tokens = buffers.params.dst_tokens.clone();
  buffers.ref_dst_positions = buffers.params.dst_positions.clone();
  buffers.ref_dst_new_cache_slots = buffers.params.dst_new_cache_slots.clone();
  buffers.ref_dst_q_seq_lens = buffers.params.dst_q_seq_lens.clone();
  buffers.ref_dst_kv_seq_lens = buffers.params.dst_kv_seq_lens.clone();
  buffers.ref_dst_q_cu_seq_lens = buffers.params.dst_q_cu_seq_lens.defined()
                                      ? buffers.params.dst_q_cu_seq_lens.clone()
                                      : torch::Tensor();
  buffers.ref_dst_linear_state_indices =
      buffers.params.dst_linear_state_indices.defined()
          ? buffers.params.dst_linear_state_indices.clone()
          : torch::Tensor();
  buffers.ref_dst_block_tables = buffers.params.dst_block_tables.clone();
  buffers.ref_dst_input_embedding =
      buffers.params.dst_input_embedding.defined()
          ? buffers.params.dst_input_embedding.clone()
          : torch::Tensor();

  buffers.ref_dst_tokens.slice(0, 0, test_case.actual_num_tokens)
      .copy_(buffers.params.src_tokens, /*non_blocking=*/true);
  buffers.ref_dst_new_cache_slots.slice(0, 0, test_case.actual_num_tokens)
      .copy_(buffers.params.src_new_cache_slots, /*non_blocking=*/true);
  if (test_case.padded_num_tokens > test_case.actual_num_tokens) {
    buffers.ref_dst_tokens
        .slice(0, test_case.actual_num_tokens, test_case.padded_num_tokens)
        .fill_(0);
    buffers.ref_dst_new_cache_slots
        .slice(0, test_case.actual_num_tokens, test_case.padded_num_tokens)
        .fill_(0);
  }

  if (test_case.with_mrope) {
    buffers.ref_dst_positions.slice(1, 0, test_case.actual_num_tokens)
        .copy_(buffers.params.src_positions, /*non_blocking=*/true);
  } else {
    buffers.ref_dst_positions.slice(0, 0, test_case.actual_num_tokens)
        .copy_(buffers.params.src_positions, /*non_blocking=*/true);
  }

  buffers.ref_dst_q_seq_lens.slice(0, 0, test_case.actual_batch_size)
      .copy_(buffers.params.src_q_seq_lens, /*non_blocking=*/true);
  buffers.ref_dst_kv_seq_lens.slice(0, 0, test_case.actual_batch_size)
      .copy_(buffers.params.src_kv_seq_lens, /*non_blocking=*/true);
  buffers.ref_dst_block_tables.slice(0, 0, test_case.actual_batch_size)
      .slice(1, 0, test_case.actual_block_table_len)
      .copy_(buffers.params.src_block_tables, /*non_blocking=*/true);

  if (test_case.with_q_cu_seq_lens) {
    buffers.ref_dst_q_cu_seq_lens.slice(0, 0, test_case.actual_batch_size)
        .copy_(buffers.params.src_q_cu_seq_lens, /*non_blocking=*/true);
  }
  if (test_case.with_linear_state_indices) {
    buffers.ref_dst_linear_state_indices
        .slice(0, 0, test_case.actual_batch_size)
        .copy_(buffers.params.src_linear_state_indices, /*non_blocking=*/true);
  }
  if (test_case.with_input_embedding) {
    buffers.ref_dst_input_embedding.slice(0, 0, test_case.actual_num_tokens)
        .copy_(buffers.params.src_input_embedding, /*non_blocking=*/true);
  }

  return buffers;
}

void run_case(const UpdaterTestCase& test_case) {
  ASSERT_TRUE(has_model_input_buffer_updater_specialization(
      test_case.embedding_dtype,
      test_case.with_mrope,
      test_case.with_input_embedding,
      test_case.with_linear_state_indices,
      test_case.with_q_cu_seq_lens));

  auto buffers = make_case(test_case);
  model_input_buffer_updater(buffers.params);
  torch::npu::synchronize();

  expect_tensor_equal(buffers.params.dst_tokens,
                      buffers.ref_dst_tokens,
                      test_case.name + ":dst_tokens");
  expect_tensor_equal(buffers.params.dst_positions,
                      buffers.ref_dst_positions,
                      test_case.name + ":dst_positions");
  expect_tensor_equal(buffers.params.dst_new_cache_slots,
                      buffers.ref_dst_new_cache_slots,
                      test_case.name + ":dst_new_cache_slots");
  expect_tensor_equal(buffers.params.dst_q_seq_lens,
                      buffers.ref_dst_q_seq_lens,
                      test_case.name + ":dst_q_seq_lens");
  expect_tensor_equal(buffers.params.dst_kv_seq_lens,
                      buffers.ref_dst_kv_seq_lens,
                      test_case.name + ":dst_kv_seq_lens");
  expect_tensor_equal(buffers.params.dst_block_tables,
                      buffers.ref_dst_block_tables,
                      test_case.name + ":dst_block_tables");
  if (test_case.with_q_cu_seq_lens) {
    expect_tensor_equal(buffers.params.dst_q_cu_seq_lens,
                        buffers.ref_dst_q_cu_seq_lens,
                        test_case.name + ":dst_q_cu_seq_lens");
  }
  if (test_case.with_linear_state_indices) {
    expect_tensor_equal(buffers.params.dst_linear_state_indices,
                        buffers.ref_dst_linear_state_indices,
                        test_case.name + ":dst_linear_state_indices");
  }
  if (test_case.with_input_embedding) {
    expect_tensor_equal(buffers.params.dst_input_embedding,
                        buffers.ref_dst_input_embedding,
                        test_case.name + ":dst_input_embedding");
  }
}

TEST_F(TileLangModelInputBufferUpdaterWrapperTest, RefCasesMatch) {
  const std::vector<UpdaterTestCase> cases = {
      {
          "plain",
          4,
          8,
          3,
          5,
          16,
          false,
          false,
          false,
          false,
          torch::kFloat32,
          0,
          11,
      },
      {
          "mrope_qcu",
          5,
          8,
          4,
          7,
          16,
          true,
          false,
          true,
          false,
          torch::kFloat32,
          0,
          12,
      },
      {
          "embedding_lsi_fp32",
          6,
          8,
          5,
          9,
          16,
          false,
          true,
          false,
          true,
          torch::kFloat32,
          96,
          13,
      },
      {
          "all_bf16",
          9,
          16,
          8,
          13,
          32,
          true,
          true,
          true,
          true,
          torch::kBFloat16,
          128,
          14,
      },
  };

  for (const auto& test_case : cases) {
    run_case(test_case);
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
