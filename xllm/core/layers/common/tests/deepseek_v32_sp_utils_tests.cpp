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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <numeric>
#include <optional>
#include <vector>

#include "core/common/global_flags.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/parallel_state/parallel_state.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/tests/tests_utils.h"
#include "layers/mlu/deepseek_v32_sp_context.h"
#include "layers/mlu/deepseek_v32_sp_metadata.h"

namespace xllm::layer::v32_sp {
namespace {

class ScriptedAllGatherProcessGroup
    : public xllm::layer::test::MockProcessGroup {
 public:
  ScriptedAllGatherProcessGroup(const torch::Device& device,
                                int64_t rank,
                                std::vector<torch::Tensor> scripted_outputs)
      : MockProcessGroup(device,
                         rank,
                         static_cast<int64_t>(scripted_outputs.size())),
        scripted_outputs_(std::move(scripted_outputs)) {}

  void allgather(const torch::Tensor& input,
                 std::vector<torch::Tensor>& outputs) override {
    (void)input;
    CHECK_EQ(outputs.size(), scripted_outputs_.size());
    for (size_t i = 0; i < scripted_outputs_.size(); ++i) {
      outputs[i].copy_(scripted_outputs_[i]);
    }
  }

  c10::intrusive_ptr<c10d::Work> allgather_async(
      const torch::Tensor& input,
      std::vector<torch::Tensor>& outputs) override {
    allgather(input, outputs);
    return xllm::layer::test::make_completed_work();
  }

 private:
  std::vector<torch::Tensor> scripted_outputs_;
};

class ScopedFlagValue {
 public:
  ScopedFlagValue(bool& flag, bool value) : bool_flag_(&flag), old_bool_(flag) {
    flag = value;
  }

  ScopedFlagValue(int32_t& flag, int32_t value)
      : int_flag_(&flag), old_int_(flag) {
    flag = value;
  }

  ~ScopedFlagValue() {
    if (bool_flag_ != nullptr) {
      *bool_flag_ = old_bool_;
    }
    if (int_flag_ != nullptr) {
      *int_flag_ = old_int_;
    }
  }

 private:
  bool* bool_flag_ = nullptr;
  int32_t* int_flag_ = nullptr;
  bool old_bool_ = false;
  int32_t old_int_ = 0;
};

AttentionMetadata make_prefill_metadata(
    const std::vector<int32_t>& q_seq_lens,
    const std::vector<int32_t>& ctx_seq_lens = {},
    const std::optional<torch::Tensor>& block_table = std::nullopt,
    BatchForwardType batch_forward_type = BatchForwardType::PREFILL) {
  AttentionMetadata attn_metadata;
  std::vector<int32_t> q_cu_seq_lens = {0};
  int32_t total = 0;
  for (int32_t q_seq_len : q_seq_lens) {
    total += q_seq_len;
    q_cu_seq_lens.push_back(total);
  }

  auto int32_options = torch::TensorOptions().dtype(torch::kInt32);
  attn_metadata.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, int32_options);
  attn_metadata.kv_cu_seq_lens = torch::tensor(q_cu_seq_lens, int32_options);
  attn_metadata.kv_seq_lens = torch::tensor(
      ctx_seq_lens.empty() ? q_seq_lens : ctx_seq_lens, int32_options);
  if (block_table.has_value()) {
    attn_metadata.block_table = *block_table;
  }
  attn_metadata.is_prefill = batch_forward_type.is_prefill();
  attn_metadata.is_chunked_prefill = batch_forward_type.is_chunked_prefill();
  attn_metadata.is_dummy = false;
  return attn_metadata;
}

torch::Tensor make_block_table(const std::vector<std::vector<int32_t>>& rows) {
  CHECK(!rows.empty());
  const int64_t row_num = static_cast<int64_t>(rows.size());
  const int64_t col_num = static_cast<int64_t>(rows.front().size());
  std::vector<int32_t> flat;
  flat.reserve(row_num * col_num);
  for (const auto& row : rows) {
    CHECK_EQ(row.size(), col_num);
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return torch::tensor(flat, torch::TensorOptions().dtype(torch::kInt32))
      .view({row_num, col_num});
}

std::vector<int32_t> extract_segment_req_idx(
    const std::vector<DeepseekV32SPSegment>& segments) {
  std::vector<int32_t> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(segment.req_idx);
  }
  return values;
}

std::vector<int32_t> extract_segment_q_tokens(
    const std::vector<DeepseekV32SPSegment>& segments) {
  std::vector<int32_t> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(segment.q_tokens);
  }
  return values;
}

std::vector<int32_t> extract_segment_suffix_k_lens(
    const std::vector<DeepseekV32SPSegment>& segments) {
  std::vector<int32_t> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(segment.suffix_k_len);
  }
  return values;
}

std::vector<int32_t> extract_segment_ctx_lens(
    const std::vector<DeepseekV32SPSegment>& segments) {
  std::vector<int32_t> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(segment.ctx_k_len);
  }
  return values;
}

TEST(DeepseekV32SPUtilsTest,
     BuildZigzagSplitPlanMatchesSingleRequestWithoutPadding) {
  const auto all_segments = build_all_sp_segments(4, {16}, {16});
  const auto runtime_artifacts =
      build_sp_runtime_artifacts(0, 4, all_segments, /*total_tokens=*/16);
  const auto& rank0 = runtime_artifacts.comm_plan;
  const auto& gathered_reorder_index =
      runtime_artifacts.gathered_reorder_index_cpu;
  const std::vector<int32_t> local_reorder_index(
      gathered_reorder_index.begin() + rank0.token_num_offset,
      gathered_reorder_index.begin() + rank0.token_num_offset +
          rank0.tokens_per_rank[0]);
  EXPECT_EQ(rank0.tokens_per_rank, (std::vector<int32_t>{4, 4, 4, 4}));
  EXPECT_EQ(rank0.padded_tokens_per_rank, (std::vector<int32_t>{4, 4, 4, 4}));
  EXPECT_EQ(local_reorder_index, (std::vector<int32_t>{0, 1, 14, 15}));
  EXPECT_EQ(gathered_reorder_index,
            (std::vector<int32_t>{
                0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9}));

  const auto segments = build_local_sp_segments(0, all_segments);
  EXPECT_EQ(extract_segment_req_idx(segments), (std::vector<int32_t>{0, 0}));
  EXPECT_EQ(extract_segment_q_tokens(segments), (std::vector<int32_t>{2, 2}));
  EXPECT_EQ(extract_segment_suffix_k_lens(segments),
            (std::vector<int32_t>{2, 16}));
  EXPECT_EQ(extract_segment_ctx_lens(segments), (std::vector<int32_t>{2, 16}));
}

TEST(DeepseekV32SPUtilsTest, BuildContextRejectsMixedBatchForwardType) {
  ScopedFlagValue enable_prefill_sp(FLAGS_enable_prefill_sp, true);
  auto attn_metadata = make_prefill_metadata({8}, {8}, std::nullopt);
  auto tokens = torch::arange(8, torch::TensorOptions().dtype(torch::kInt64));
  xllm::layer::test::MockProcessGroup process_group(torch::kCPU,
                                                    /*rank=*/0,
                                                    /*world_size=*/2);

  auto context = build_deepseek_v32_sp_context(attn_metadata,
                                               BatchForwardType::MIXED,
                                               tokens,
                                               &process_group,
                                               /*curr_rank=*/0,
                                               /*world_size=*/2);

  EXPECT_FALSE(context.has_value());
}

TEST(DeepseekV32SPUtilsTest,
     BuildZigzagSplitPlanMatchesSingleRequestWithPadding) {
  const auto all_segments = build_all_sp_segments(4, {10}, {10});
  const auto runtime_artifacts =
      build_sp_runtime_artifacts(2, 4, all_segments, /*total_tokens=*/10);
  const auto& rank2 = runtime_artifacts.comm_plan;
  const auto& gathered_reorder_index =
      runtime_artifacts.gathered_reorder_index_cpu;
  const std::vector<int32_t> local_reorder_index(
      gathered_reorder_index.begin() + rank2.token_num_offset,
      gathered_reorder_index.begin() + rank2.token_num_offset +
          rank2.tokens_per_rank[2]);
  EXPECT_EQ(rank2.tokens_per_rank, (std::vector<int32_t>{3, 3, 2, 2}));
  EXPECT_EQ(rank2.padded_tokens_per_rank, (std::vector<int32_t>{3, 3, 3, 3}));
  EXPECT_EQ(local_reorder_index, (std::vector<int32_t>{4, 7}));

  const auto segments = build_local_sp_segments(2, all_segments);
  EXPECT_EQ(extract_segment_req_idx(segments), (std::vector<int32_t>{0, 0}));
  EXPECT_EQ(extract_segment_q_tokens(segments), (std::vector<int32_t>{1, 1}));
  EXPECT_EQ(extract_segment_suffix_k_lens(segments),
            (std::vector<int32_t>{5, 8}));
  EXPECT_EQ(extract_segment_ctx_lens(segments), (std::vector<int32_t>{5, 8}));
}

TEST(DeepseekV32SPUtilsTest, BuildZigzagSplitPlanTracksContextLens) {
  AttentionMetadata attn_metadata = make_prefill_metadata({4, 6}, {16, 22});

  EXPECT_EQ(extract_q_seq_lens(attn_metadata), (std::vector<int32_t>{4, 6}));
  EXPECT_EQ(extract_ctx_seq_lens(attn_metadata),
            (std::vector<int32_t>{16, 22}));

  const auto all_segments = build_all_sp_segments(4, {4, 6}, {16, 22});
  const auto segments = build_local_sp_segments(1, all_segments);
  EXPECT_EQ(extract_segment_q_tokens(segments),
            (std::vector<int32_t>{1, 0, 1, 1}));
  EXPECT_EQ(extract_segment_suffix_k_lens(segments),
            (std::vector<int32_t>{2, 0, 2, 5}));
  EXPECT_EQ(extract_segment_ctx_lens(segments),
            (std::vector<int32_t>{14, 12, 18, 21}));
}

TEST(DeepseekV32SPUtilsTest,
     BuildDeepseekV32SPContextBuildsMetadataForCurrentRank) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue world_size_flag(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata =
      make_prefill_metadata({4, 6, 11}, {12, 16, 32});
  torch::Tensor tokens =
      torch::arange(0, 21, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    1,
                                    4);

  ASSERT_TRUE(maybe_context.has_value());
  const auto& context = maybe_context.value();
  EXPECT_EQ(context.total_tokens, 21);
  EXPECT_EQ(context.rank, 1);
  EXPECT_EQ(context.comm_plan.tokens_per_rank,
            (std::vector<int32_t>{6, 6, 5, 4}));
  EXPECT_EQ(context.comm_plan.padded_tokens_per_rank,
            (std::vector<int32_t>{6, 6, 6, 6}));
  EXPECT_EQ(context.comm_plan.token_num_offset, 6);

  EXPECT_TRUE(
      torch::equal(reorder_to_local_shard(tokens, context),
                   torch::tensor({1, 5, 8, 12, 13, 19},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(context.gathered_reorder_index,
                   torch::tensor({0,  4, 9, 10, 11, 20, 1, 5, 8,  12, 13,
                                  19, 2, 6, 14, 15, 18, 3, 7, 16, 17},
                                 torch::TensorOptions().dtype(torch::kInt64))));

  EXPECT_TRUE(
      torch::equal(context.local_attn_metadata.q_cu_seq_lens,
                   torch::tensor({0, 1, 1, 2, 3, 5, 6},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(context.local_attn_metadata.kv_seq_lens,
                   torch::tensor({2, 0, 2, 5, 4, 10},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_EQ(context.local_attn_metadata.max_query_len, 2);
  EXPECT_EQ(context.local_attn_metadata.max_seq_len, 10);
  EXPECT_EQ(context.sp_meta.k_pack_starts_cpu,
            (std::vector<int32_t>{0, 4, 4, 10, 10}));
  EXPECT_EQ(context.sp_meta.k_pack_lens_cpu,
            (std::vector<int32_t>{2, 2, 5, 4, 10}));
  EXPECT_TRUE(
      torch::equal(context.sp_meta.seg_q_cu_lens,
                   torch::tensor({0, 1, 1, 2, 3, 5, 6},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(context.sp_meta.seg_suffix_k_cu_lens,
                   torch::tensor({0, 2, 2, 4, 9, 13, 23},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(context.sp_meta.seg_ctx_lens,
                   torch::tensor({10, 8, 12, 15, 25, 31},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_FALSE(context.sp_meta.seg_block_table.defined());
}

TEST(DeepseekV32SPUtilsTest,
     BuildDeepseekV32SPContextFallsBackWhenSeqShorterThanWorldSize) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue nnodes_flag(FLAGS_nnodes, 2);

  AttentionMetadata attn_metadata = make_prefill_metadata({3});
  torch::Tensor tokens =
      torch::arange(0, 3, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    0,
                                    4);

  EXPECT_FALSE(maybe_context.has_value());
}

TEST(DeepseekV32SPUtilsTest,
     BuildDeepseekV32SPContextAcceptsChunkedPrefillBatch) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue world_size_flag(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata(
      {4, 6}, {16, 22}, std::nullopt, BatchForwardType::CHUNKED_PREFILL);
  torch::Tensor tokens =
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::CHUNKED_PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    1,
                                    4);

  ASSERT_TRUE(maybe_context.has_value());
  const auto& context = maybe_context.value();
  EXPECT_TRUE(
      torch::equal(context.local_attn_metadata.kv_seq_lens,
                   torch::tensor({2, 0, 2, 5},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(context.sp_meta.seg_ctx_lens,
                   torch::tensor({14, 12, 18, 21},
                                 torch::TensorOptions().dtype(torch::kInt32))));
}

TEST(DeepseekV32SPUtilsTest, BuildDeepseekV32SPContextRejectsMixedBatch) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue world_size_flag(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata({4, 6}, {16, 22});
  torch::Tensor tokens =
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::MIXED,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    1,
                                    4);

  EXPECT_FALSE(maybe_context.has_value());
}

TEST(DeepseekV32SPUtilsTest, BuildSPMetadataTracksSegmentView) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue world_size_flag(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata(
      {4, 6, 11},
      {12, 16, 32},
      make_block_table({{10, 11, 12}, {20, 21, 22}, {30, 31, 32}}));
  torch::Tensor tokens =
      torch::arange(0, 21, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    1,
                                    4);
  ASSERT_TRUE(maybe_context.has_value());
  const auto& sp_meta = maybe_context->sp_meta;

  EXPECT_EQ(sp_meta.k_pack_starts_cpu, (std::vector<int32_t>{0, 4, 4, 10, 10}));
  EXPECT_EQ(sp_meta.k_pack_lens_cpu, (std::vector<int32_t>{2, 2, 5, 4, 10}));
  EXPECT_EQ(sp_meta.k_ctx_pack_starts_cpu,
            (std::vector<int32_t>{0, 0, 12, 12, 28, 28}));
  EXPECT_EQ(sp_meta.k_ctx_pack_lens_cpu,
            (std::vector<int32_t>{10, 8, 12, 15, 25, 31}));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_q_cu_lens,
                   torch::tensor({0, 1, 1, 2, 3, 5, 6},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_suffix_k_cu_lens,
                   torch::tensor({0, 2, 2, 4, 9, 13, 23},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_ctx_cu_lens,
                   torch::tensor({0, 10, 18, 30, 45, 70, 101},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_ctx_lens,
                   torch::tensor({10, 8, 12, 15, 25, 31},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_block_table,
                   torch::tensor({{10, 11, 12},
                                  {10, 11, 12},
                                  {20, 21, 22},
                                  {20, 21, 22},
                                  {30, 31, 32},
                                  {30, 31, 32}},
                                 torch::TensorOptions().dtype(torch::kInt32))));
}

TEST(DeepseekV32SPUtilsTest, BuildSPMetadataTracksPartialPrefixHit) {
  AttentionMetadata attn_metadata = make_prefill_metadata({4, 6}, {16, 22});
  const auto all_segments = build_all_sp_segments(4, {4, 6}, {16, 22});
  const auto local_segments = build_local_sp_segments(1, all_segments);
  const auto local_attn_metadata =
      build_local_prefill_attention_metadata(attn_metadata, local_segments);
  const auto sp_meta = build_sp_metadata(attn_metadata, local_segments, {4, 6});

  EXPECT_TRUE(
      torch::equal(local_attn_metadata.kv_seq_lens,
                   torch::tensor({2, 0, 2, 5},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_EQ(local_attn_metadata.max_seq_len, 5);
  EXPECT_EQ(sp_meta.k_pack_starts_cpu, (std::vector<int32_t>{0, 4, 4}));
  EXPECT_EQ(sp_meta.k_pack_lens_cpu, (std::vector<int32_t>{2, 2, 5}));
  EXPECT_EQ(sp_meta.k_ctx_pack_starts_cpu,
            (std::vector<int32_t>{0, 0, 16, 16}));
  EXPECT_EQ(sp_meta.k_ctx_pack_lens_cpu,
            (std::vector<int32_t>{14, 12, 18, 21}));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_ctx_lens,
                   torch::tensor({14, 12, 18, 21},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_suffix_k_cu_lens,
                   torch::tensor({0, 2, 2, 4, 9},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_ctx_cu_lens,
                   torch::tensor({0, 14, 26, 44, 65},
                                 torch::TensorOptions().dtype(torch::kInt32))));
}

TEST(DeepseekV32SPUtilsTest,
     BuildLocalPrefillMetadataKeepsSuffixOnlyLensForChunkedPrefill) {
  AttentionMetadata attn_metadata = make_prefill_metadata(
      {4, 6}, {16, 22}, std::nullopt, BatchForwardType::CHUNKED_PREFILL);
  const auto all_segments = build_all_sp_segments(4, {4, 6}, {16, 22});
  const auto local_segments = build_local_sp_segments(1, all_segments);
  const auto local_attn_metadata =
      build_local_prefill_attention_metadata(attn_metadata, local_segments);

  EXPECT_TRUE(
      torch::equal(local_attn_metadata.q_cu_seq_lens,
                   torch::tensor({0, 1, 1, 2, 3},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(local_attn_metadata.kv_cu_seq_lens,
                   torch::tensor({0, 2, 2, 4, 9},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(local_attn_metadata.kv_seq_lens,
                   torch::tensor({2, 0, 2, 5},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_EQ(local_attn_metadata.max_query_len, 1);
  EXPECT_EQ(local_attn_metadata.max_seq_len, 5);
  EXPECT_FALSE(
      torch::equal(local_attn_metadata.kv_seq_lens,
                   torch::tensor({14, 12, 18, 21},
                                 torch::TensorOptions().dtype(torch::kInt32))));
}

TEST(DeepseekV32SPUtilsTest,
     BuildSPMetadataKeepsChunkedBlockTableForIndexerSelect) {
  AttentionMetadata attn_metadata = make_prefill_metadata(
      {4, 6},
      {16, 22},
      make_block_table({{10, 11, 12, 13}, {20, 21, 22, 23}}),
      BatchForwardType::CHUNKED_PREFILL);
  const auto all_segments = build_all_sp_segments(4, {4, 6}, {16, 22});
  const auto local_segments = build_local_sp_segments(1, all_segments);
  const auto sp_meta = build_sp_metadata(attn_metadata, local_segments, {4, 6});

  EXPECT_EQ(sp_meta.k_pack_lens_cpu, (std::vector<int32_t>{2, 2, 5}));
  EXPECT_EQ(sp_meta.k_ctx_pack_starts_cpu,
            (std::vector<int32_t>{0, 0, 16, 16}));
  EXPECT_EQ(sp_meta.k_ctx_pack_lens_cpu,
            (std::vector<int32_t>{14, 12, 18, 21}));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_suffix_k_cu_lens,
                   torch::tensor({0, 2, 2, 4, 9},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_ctx_cu_lens,
                   torch::tensor({0, 14, 26, 44, 65},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_ctx_lens,
                   torch::tensor({14, 12, 18, 21},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_block_table,
                   torch::tensor({{10, 11, 12, 13},
                                  {10, 11, 12, 13},
                                  {20, 21, 22, 23},
                                  {20, 21, 22, 23}},
                                 torch::TensorOptions().dtype(torch::kInt32))));
}

TEST(DeepseekV32SPUtilsTest, BuildSPContextKeepsExactBlockRollbackLens) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue world_size_flag(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata(
      {4, 8}, {20, 24}, make_block_table({{100, 101, 102}, {200, 201, 202}}));
  torch::Tensor tokens =
      torch::arange(0, 12, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    0,
                                    4);

  ASSERT_TRUE(maybe_context.has_value());
  const auto& context = maybe_context.value();
  EXPECT_EQ(context.local_attn_metadata.q_cu_seq_lens.size(0), 5);
  EXPECT_TRUE(
      torch::equal(context.sp_meta.seg_ctx_lens,
                   torch::tensor({17, 16, 17, 24},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(torch::equal(
      context.sp_meta.seg_block_table,
      torch::tensor(
          {{100, 101, 102}, {100, 101, 102}, {200, 201, 202}, {200, 201, 202}},
          torch::TensorOptions().dtype(torch::kInt32))));
}

TEST(DeepseekV32SPUtilsTest, BuildSPMetadataTracksMixedHitMiss) {
  AttentionMetadata attn_metadata = make_prefill_metadata(
      {8, 8}, {8, 20}, make_block_table({{1, 2, 3}, {4, 5, 6}}));
  const auto all_segments = build_all_sp_segments(4, {8, 8}, {8, 20});
  const auto local_segments = build_local_sp_segments(2, all_segments);
  const auto sp_meta = build_sp_metadata(attn_metadata, local_segments, {8, 8});

  EXPECT_EQ(extract_segment_q_tokens(local_segments),
            (std::vector<int32_t>{1, 1, 1, 1}));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_suffix_k_cu_lens,
                   torch::tensor({0, 3, 9, 12, 18},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_ctx_lens,
                   torch::tensor({3, 6, 15, 18},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_block_table,
                   torch::tensor({{1, 2, 3}, {1, 2, 3}, {4, 5, 6}, {4, 5, 6}},
                                 torch::TensorOptions().dtype(torch::kInt32))));
}

TEST(DeepseekV32SPUtilsTest, PackSPKForIndexerExpandsSegmentPrefixes) {
  DeepseekV32SPMetadata sp_meta;
  sp_meta.k_pack_starts_cpu = {0, 0, 4, 4};
  sp_meta.k_pack_lens_cpu = {2, 4, 1, 3};

  torch::Tensor k_global =
      torch::arange(0, 7, torch::TensorOptions().dtype(torch::kFloat32))
          .view({7, 1});
  torch::Tensor k_packed = pack_sp_k_for_indexer(k_global, sp_meta);

  EXPECT_TRUE(torch::equal(
      k_packed,
      torch::tensor({{0.0f},
                     {1.0f},
                     {0.0f},
                     {1.0f},
                     {2.0f},
                     {3.0f},
                     {4.0f},
                     {4.0f},
                     {5.0f},
                     {6.0f}},
                    torch::TensorOptions().dtype(torch::kFloat32))));
}

TEST(DeepseekV32SPUtilsTest, PackSPCtxKExpandsSegmentContexts) {
  DeepseekV32SPMetadata sp_meta;
  sp_meta.k_ctx_pack_starts_cpu = {0, 0, 5};
  sp_meta.k_ctx_pack_lens_cpu = {3, 5, 2};

  torch::Tensor k_ctx =
      torch::arange(0, 7, torch::TensorOptions().dtype(torch::kFloat32))
          .view({7, 1});
  torch::Tensor k_packed = pack_sp_ctx_k(k_ctx, sp_meta);

  EXPECT_TRUE(torch::equal(
      k_packed,
      torch::tensor({{0.0f},
                     {1.0f},
                     {2.0f},
                     {0.0f},
                     {1.0f},
                     {2.0f},
                     {3.0f},
                     {4.0f},
                     {5.0f},
                     {6.0f}},
                    torch::TensorOptions().dtype(torch::kFloat32))));
}

TEST(DeepseekV32SPUtilsTest, SliceLocalPackedUsesRankOffset) {
  DeepseekV32SPContext context;
  context.rank = 1;
  context.comm_plan.tokens_per_rank = {2, 3, 1};
  context.comm_plan.token_num_offset = 2;

  torch::Tensor packed = torch::tensor(
      {10, 11, 20, 21, 22, 30}, torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor local = slice_local_packed(packed, context);

  EXPECT_TRUE(
      torch::equal(local,
                   torch::tensor({20, 21, 22},
                                 torch::TensorOptions().dtype(torch::kInt32))));
}

TEST(DeepseekV32SPUtilsTest,
     BuildDeepseekV32SPContextReturnsNulloptWhenDisabled) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, false);
  ScopedFlagValue world_size_flag(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata({8});
  torch::Tensor tokens =
      torch::arange(0, 8, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    0,
                                    4);

  EXPECT_FALSE(maybe_context.has_value());
}

TEST(DeepseekV32SPUtilsTest, ReorderByIndexSlicesFirstDimension) {
  torch::Tensor hidden_states = torch::tensor(
      {{0.0f, 10.0f}, {1.0f, 11.0f}, {2.0f, 12.0f}, {3.0f, 13.0f}},
      torch::TensorOptions().dtype(torch::kFloat32));
  torch::Tensor token_index =
      torch::tensor({3, 1}, torch::TensorOptions().dtype(torch::kInt32));

  torch::Tensor reordered = reorder_by_index(hidden_states, token_index);

  EXPECT_TRUE(torch::equal(
      reordered,
      torch::tensor({{3.0f, 13.0f}, {1.0f, 11.0f}},
                    torch::TensorOptions().dtype(torch::kFloat32))));
}

TEST(DeepseekV32SPUtilsTest, RestoreGatheredToGlobalOrderWithoutPadding) {
  ScopedFlagValue use_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue nnodes(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata({16});
  torch::Tensor tokens =
      torch::arange(0, 16, torch::TensorOptions().dtype(torch::kInt32));
  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    0,
                                    4);
  ASSERT_TRUE(maybe_context.has_value());
  const auto& context = maybe_context.value();

  torch::Tensor gathered =
      context.gathered_reorder_index.to(torch::kFloat32).contiguous();
  torch::Tensor restored = restore_gathered_to_global_order(
      gathered, context, GatheredTensorLayout::kPacked);

  EXPECT_TRUE(torch::equal(
      restored,
      torch::arange(0, 16, torch::TensorOptions().dtype(torch::kFloat32))));
}

TEST(DeepseekV32SPUtilsTest, RestoreGatheredToGlobalOrderDropsPadding) {
  ScopedFlagValue use_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue nnodes(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata({10});
  torch::Tensor tokens =
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kInt32));
  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    1,
                                    4);
  ASSERT_TRUE(maybe_context.has_value());
  const auto& context = maybe_context.value();

  const int64_t padded_token_num =
      std::accumulate(context.comm_plan.padded_tokens_per_rank.begin(),
                      context.comm_plan.padded_tokens_per_rank.end(),
                      int64_t{0});
  torch::Tensor gathered = torch::zeros(
      {padded_token_num}, torch::TensorOptions().dtype(torch::kFloat32));
  auto* gathered_ptr = gathered.data_ptr<float>();
  auto gathered_index = context.gathered_reorder_index.to(torch::kCPU);
  const auto* gathered_index_ptr = gathered_index.data_ptr<int64_t>();
  int64_t padded_offset = 0;
  int64_t packed_offset = 0;
  for (size_t rank = 0; rank < context.comm_plan.tokens_per_rank.size();
       ++rank) {
    const int32_t valid_token_num = context.comm_plan.tokens_per_rank[rank];
    for (int32_t i = 0; i < valid_token_num; ++i) {
      gathered_ptr[padded_offset + i] =
          static_cast<float>(gathered_index_ptr[packed_offset + i]);
    }
    padded_offset += context.comm_plan.padded_tokens_per_rank[rank];
    packed_offset += valid_token_num;
  }

  torch::Tensor restored = restore_gathered_to_global_order(
      gathered, context, GatheredTensorLayout::kPaddedPacked);

  EXPECT_TRUE(torch::equal(
      restored,
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32))));
}

TEST(DeepseekV32SPUtilsTest, AllGatherAcrossRanksRestoresGlobalOrder) {
  ScopedFlagValue use_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue nnodes(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata({10});
  torch::Tensor tokens =
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kInt32));
  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    1,
                                    4);
  ASSERT_TRUE(maybe_context.has_value());
  auto context = maybe_context.value();

  const int64_t padded_token_num = context.comm_plan.padded_tokens_per_rank[0];
  std::vector<torch::Tensor> scripted_outputs;
  scripted_outputs.reserve(4);
  auto across_world =
      context.gathered_reorder_index.to(torch::kCPU).contiguous();
  const auto* across_world_ptr = across_world.data_ptr<int64_t>();
  int32_t token_offset = 0;
  for (int64_t world_rank = 0; world_rank < 4; ++world_rank) {
    std::vector<float> values(padded_token_num, -1.0f);
    const int32_t valid_token_num =
        context.comm_plan.tokens_per_rank[world_rank];
    for (int32_t i = 0; i < valid_token_num; ++i) {
      values[i] = static_cast<float>(across_world_ptr[token_offset + i]);
    }
    token_offset += valid_token_num;
    scripted_outputs.push_back(
        torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32)));
  }

  ScriptedAllGatherProcessGroup scripted_group(
      torch::Device(torch::kCPU), 1, std::move(scripted_outputs));
  context.process_group = &scripted_group;
  torch::Tensor local_tensor = torch::tensor(
      {2.0f, 3.0f, 8.0f}, torch::TensorOptions().dtype(torch::kFloat32));

  torch::Tensor gathered = all_gather_across_ranks(local_tensor, context);
  torch::Tensor restored = restore_gathered_to_global_order(
      gathered, context, GatheredTensorLayout::kPacked);

  EXPECT_TRUE(torch::equal(
      restored,
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32))));
}

TEST(DeepseekV32SPUtilsTest, GatherSupportsUnevenThreeDimensionalInputs) {
  std::vector<torch::Tensor> scripted_outputs;
  scripted_outputs.reserve(4);
  for (int64_t world_rank = 0; world_rank < 4; ++world_rank) {
    scripted_outputs.push_back(
        torch::full({3, 2, 2},
                    static_cast<float>(world_rank),
                    torch::TensorOptions().dtype(torch::kFloat32)));
  }

  ScriptedAllGatherProcessGroup process_group(
      torch::Device(torch::kCPU), 1, std::move(scripted_outputs));
  torch::Tensor local_tensor = torch::full(
      {2, 2, 2}, 1.0f, torch::TensorOptions().dtype(torch::kFloat32));

  torch::Tensor gathered = xllm::parallel_state::gather(
      local_tensor, &process_group, std::vector<int32_t>{3, 2, 3, 1});

  EXPECT_EQ(gathered.sizes(), (torch::IntArrayRef{9, 2, 2}));
  EXPECT_TRUE(torch::allclose(gathered[0], torch::zeros({2, 2})));
  EXPECT_TRUE(torch::allclose(gathered[2], torch::zeros({2, 2})));
  EXPECT_TRUE(torch::allclose(gathered[3], torch::ones({2, 2})));
  EXPECT_TRUE(torch::allclose(gathered[4], torch::ones({2, 2})));
  EXPECT_TRUE(torch::allclose(gathered[5], torch::full({2, 2}, 2.0f)));
  EXPECT_TRUE(torch::allclose(gathered[7], torch::full({2, 2}, 2.0f)));
  EXPECT_TRUE(torch::allclose(gathered[8], torch::full({2, 2}, 3.0f)));
}

TEST(DeepseekV32SPUtilsTest, LaunchAndFinishGatherMatchBlockingGather) {
  std::vector<torch::Tensor> scripted_outputs;
  scripted_outputs.reserve(4);
  for (int64_t world_rank = 0; world_rank < 4; ++world_rank) {
    scripted_outputs.push_back(
        torch::full({3, 2},
                    static_cast<float>(world_rank),
                    torch::TensorOptions().dtype(torch::kFloat32)));
  }

  ScriptedAllGatherProcessGroup process_group(
      torch::Device(torch::kCPU), 1, std::move(scripted_outputs));
  torch::Tensor local_tensor =
      torch::full({2, 2}, 1.0f, torch::TensorOptions().dtype(torch::kFloat32));

  auto gather_ctx = xllm::parallel_state::launch_gather(
      local_tensor, &process_group, std::vector<int32_t>{3, 2, 3, 1});
  torch::Tensor gathered = xllm::parallel_state::finish_gather(gather_ctx);
  auto float_options = torch::TensorOptions().dtype(torch::kFloat32);

  EXPECT_EQ(gathered.sizes(), (torch::IntArrayRef{9, 2}));
  EXPECT_TRUE(torch::allclose(gathered[0], torch::zeros({2}, float_options)));
  EXPECT_TRUE(torch::allclose(gathered[3], torch::ones({2}, float_options)));
  EXPECT_TRUE(
      torch::allclose(gathered[5], torch::full({2}, 2.0f, float_options)));
  EXPECT_TRUE(
      torch::allclose(gathered[8], torch::full({2}, 3.0f, float_options)));
}

TEST(DeepseekV32SPUtilsTest, PaddedGatherRestoresGlobalOrder) {
  ScopedFlagValue use_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue nnodes(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata({10});
  torch::Tensor tokens =
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kInt32));
  auto maybe_context =
      build_deepseek_v32_sp_context(attn_metadata,
                                    BatchForwardType::PREFILL,
                                    tokens,
                                    reinterpret_cast<ProcessGroup*>(0x1),
                                    1,
                                    4);
  ASSERT_TRUE(maybe_context.has_value());
  auto context = maybe_context.value();

  const int64_t padded_token_num = context.comm_plan.padded_tokens_per_rank[0];
  std::vector<torch::Tensor> scripted_outputs;
  scripted_outputs.reserve(4);
  auto across_world =
      context.gathered_reorder_index.to(torch::kCPU).contiguous();
  const auto* across_world_ptr = across_world.data_ptr<int64_t>();
  int32_t token_offset = 0;
  for (int64_t world_rank = 0; world_rank < 4; ++world_rank) {
    std::vector<float> values(padded_token_num, -1.0f);
    const int32_t valid_token_num =
        context.comm_plan.tokens_per_rank[world_rank];
    for (int32_t i = 0; i < valid_token_num; ++i) {
      values[i] = static_cast<float>(across_world_ptr[token_offset + i]);
    }
    token_offset += valid_token_num;
    scripted_outputs.push_back(
        torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32)));
  }

  ScriptedAllGatherProcessGroup scripted_group(
      torch::Device(torch::kCPU), 1, std::move(scripted_outputs));
  context.process_group = &scripted_group;
  torch::Tensor local_tensor = torch::tensor(
      {2.0f, 3.0f, 8.0f}, torch::TensorOptions().dtype(torch::kFloat32));

  auto gather_handle =
      launch_gather_padded(pad_to_sp_rows(local_tensor, context), context);
  torch::Tensor restored = finish_gather_padded(gather_handle, context);

  EXPECT_TRUE(torch::equal(
      restored,
      torch::arange(0, 10, torch::TensorOptions().dtype(torch::kFloat32))));
}

}  // namespace
}  // namespace xllm::layer::v32_sp
