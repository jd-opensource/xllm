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

#include <vector>

#include "core/common/global_flags.h"
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

AttentionMetadata make_prefill_metadata(const std::vector<int32_t>& seq_lens) {
  AttentionMetadata attn_metadata;
  std::vector<int32_t> q_cu_seq_lens = {0};
  int32_t total = 0;
  for (int32_t seq_len : seq_lens) {
    total += seq_len;
    q_cu_seq_lens.push_back(total);
  }

  auto int32_options = torch::TensorOptions().dtype(torch::kInt32);
  attn_metadata.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, int32_options);
  attn_metadata.kv_cu_seq_lens = torch::tensor(q_cu_seq_lens, int32_options);
  attn_metadata.kv_seq_lens = torch::tensor(seq_lens, int32_options);
  attn_metadata.is_prefill = true;
  attn_metadata.is_chunked_prefill = false;
  attn_metadata.is_dummy = false;
  return attn_metadata;
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

std::vector<int32_t> extract_segment_q_offsets(
    const std::vector<DeepseekV32SPSegment>& segments) {
  std::vector<int32_t> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(segment.q_offset);
  }
  return values;
}

std::vector<int32_t> extract_segment_k_lens(
    const std::vector<DeepseekV32SPSegment>& segments) {
  std::vector<int32_t> values;
  values.reserve(segments.size());
  for (const auto& segment : segments) {
    values.push_back(segment.k_len);
  }
  return values;
}

TEST(DeepseekV32SPUtilsTest,
     BuildZigzagSplitPlanMatchesSingleRequestWithoutPadding) {
  const auto all_segments = build_all_sp_segments(4, {16});
  const DeepseekV32SPCommPlan rank0 =
      build_zigzag_comm_plan(0, 4, all_segments, /*total_tokens=*/16);
  EXPECT_EQ(rank0.tokens_per_rank, (std::vector<int32_t>{4, 4, 4, 4}));
  EXPECT_EQ(rank0.padded_tokens_per_rank, (std::vector<int32_t>{4, 4, 4, 4}));
  EXPECT_EQ(rank0.local_reorder_index_cpu,
            (std::vector<int32_t>{0, 1, 14, 15}));
  EXPECT_EQ(rank0.gathered_reorder_index_cpu,
            (std::vector<int32_t>{
                0, 1, 14, 15, 2, 3, 12, 13, 4, 5, 10, 11, 6, 7, 8, 9}));

  const auto segments = build_local_sp_segments(0, all_segments);
  EXPECT_EQ(extract_segment_req_idx(segments), (std::vector<int32_t>{0, 0}));
  EXPECT_EQ(extract_segment_q_tokens(segments), (std::vector<int32_t>{2, 2}));
  EXPECT_EQ(extract_segment_q_offsets(segments), (std::vector<int32_t>{0, 14}));
  EXPECT_EQ(extract_segment_k_lens(segments), (std::vector<int32_t>{2, 16}));
}

TEST(DeepseekV32SPUtilsTest,
     BuildZigzagSplitPlanMatchesSingleRequestWithPadding) {
  const auto all_segments = build_all_sp_segments(4, {10});
  const DeepseekV32SPCommPlan rank2 =
      build_zigzag_comm_plan(2, 4, all_segments, /*total_tokens=*/10);
  EXPECT_EQ(rank2.tokens_per_rank, (std::vector<int32_t>{3, 3, 2, 2}));
  EXPECT_EQ(rank2.padded_tokens_per_rank, (std::vector<int32_t>{3, 3, 3, 3}));
  EXPECT_EQ(rank2.local_reorder_index_cpu, (std::vector<int32_t>{4, 7}));

  const auto segments = build_local_sp_segments(2, all_segments);
  EXPECT_EQ(extract_segment_req_idx(segments), (std::vector<int32_t>{0, 0}));
  EXPECT_EQ(extract_segment_q_tokens(segments), (std::vector<int32_t>{1, 1}));
  EXPECT_EQ(extract_segment_q_offsets(segments), (std::vector<int32_t>{4, 7}));
  EXPECT_EQ(extract_segment_k_lens(segments), (std::vector<int32_t>{5, 8}));
}

TEST(DeepseekV32SPUtilsTest,
     BuildDeepseekV32SPContextBuildsMetadataForCurrentRank) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue world_size_flag(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata({4, 6, 11});
  torch::Tensor tokens =
      torch::arange(0, 21, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context = build_deepseek_v32_sp_context(
      attn_metadata, tokens, reinterpret_cast<ProcessGroup*>(0x1), 1, 4);

  ASSERT_TRUE(maybe_context.has_value());
  const auto& context = maybe_context.value();
  EXPECT_EQ(context.total_tokens, 21);
  EXPECT_EQ(context.rank, 1);
  EXPECT_EQ(context.world_size, 4);
  EXPECT_EQ(extract_segment_req_idx(context.segments),
            (std::vector<int32_t>{0, 0, 1, 1, 2, 2}));
  EXPECT_EQ(extract_segment_q_tokens(context.segments),
            (std::vector<int32_t>{1, 0, 1, 1, 2, 1}));
  EXPECT_EQ(extract_segment_q_offsets(context.segments),
            (std::vector<int32_t>{1, 4, 1, 4, 2, 9}));
  EXPECT_EQ(extract_segment_k_lens(context.segments),
            (std::vector<int32_t>{2, 0, 2, 5, 4, 10}));
  EXPECT_EQ(context.comm_plan.tokens_per_rank,
            (std::vector<int32_t>{6, 6, 5, 4}));
  EXPECT_EQ(context.comm_plan.padded_tokens_per_rank,
            (std::vector<int32_t>{6, 6, 6, 6}));
  EXPECT_EQ(context.comm_plan.token_num_offset, 6);

  EXPECT_TRUE(
      torch::equal(context.local_reorder_index,
                   torch::tensor({1, 5, 8, 12, 13, 19},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(context.gathered_reorder_index,
                   torch::tensor({0,  4, 9, 10, 11, 20, 1, 5, 8,  12, 13,
                                  19, 2, 6, 14, 15, 18, 3, 7, 16, 17},
                                 torch::TensorOptions().dtype(torch::kInt64))));
  EXPECT_TRUE(
      torch::equal(context.gathered_padded_reorder_index,
                   torch::tensor({0, 4, 9,  10, 11, 20, 1, 5, 8,  12, 13, 19,
                                  2, 6, 14, 15, 18, 21, 3, 7, 16, 17, 21, 21},
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
  EXPECT_EQ(context.sp_meta.req_offsets_cpu, (std::vector<int32_t>{0, 4, 10}));
  EXPECT_EQ(context.sp_meta.segments.size(), 6);
  EXPECT_TRUE(
      torch::equal(context.sp_meta.seg_q_cu_lens,
                   torch::tensor({0, 1, 1, 2, 3, 5, 6},
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

  auto maybe_context = build_deepseek_v32_sp_context(
      attn_metadata, tokens, reinterpret_cast<ProcessGroup*>(0x1), 0, 4);

  EXPECT_FALSE(maybe_context.has_value());
}

TEST(DeepseekV32SPUtilsTest, BuildSPMetadataTracksSegmentView) {
  ScopedFlagValue enable_sp(FLAGS_enable_prefill_sp, true);
  ScopedFlagValue world_size_flag(FLAGS_nnodes, 4);

  AttentionMetadata attn_metadata = make_prefill_metadata({4, 6, 11});
  attn_metadata.block_table =
      torch::tensor({{10, 11, 12}, {20, 21, 22}, {30, 31, 32}},
                    torch::TensorOptions().dtype(torch::kInt32));
  torch::Tensor tokens =
      torch::arange(0, 21, torch::TensorOptions().dtype(torch::kInt32));

  auto maybe_context = build_deepseek_v32_sp_context(
      attn_metadata, tokens, reinterpret_cast<ProcessGroup*>(0x1), 1, 4);
  ASSERT_TRUE(maybe_context.has_value());
  const auto& sp_meta = maybe_context->sp_meta;

  EXPECT_EQ(sp_meta.req_offsets_cpu, (std::vector<int32_t>{0, 4, 10}));
  EXPECT_EQ(extract_segment_req_idx(sp_meta.segments),
            (std::vector<int32_t>{0, 0, 1, 1, 2, 2}));
  EXPECT_EQ(extract_segment_q_tokens(sp_meta.segments),
            (std::vector<int32_t>{1, 0, 1, 1, 2, 1}));
  EXPECT_EQ(extract_segment_k_lens(sp_meta.segments),
            (std::vector<int32_t>{2, 0, 2, 5, 4, 10}));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_q_cu_lens,
                   torch::tensor({0, 1, 1, 2, 3, 5, 6},
                                 torch::TensorOptions().dtype(torch::kInt32))));
  EXPECT_TRUE(
      torch::equal(sp_meta.seg_k_cu_lens,
                   torch::tensor({0, 2, 2, 4, 9, 13, 23},
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

TEST(DeepseekV32SPUtilsTest, PackSPKForIndexerExpandsSegmentPrefixes) {
  DeepseekV32SPMetadata sp_meta;
  sp_meta.req_offsets_cpu = {0, 4};
  sp_meta.segments = {
      {.req_idx = 0, .rank = 0, .q_tokens = 0, .q_offset = 0, .k_len = 2},
      {.req_idx = 0, .rank = 0, .q_tokens = 0, .q_offset = 0, .k_len = 4},
      {.req_idx = 1, .rank = 0, .q_tokens = 0, .q_offset = 0, .k_len = 1},
      {.req_idx = 1, .rank = 0, .q_tokens = 0, .q_offset = 0, .k_len = 3},
  };

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

  auto maybe_context = build_deepseek_v32_sp_context(
      attn_metadata, tokens, reinterpret_cast<ProcessGroup*>(0x1), 0, 4);

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
  auto maybe_context = build_deepseek_v32_sp_context(
      attn_metadata, tokens, reinterpret_cast<ProcessGroup*>(0x1), 0, 4);
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
  auto maybe_context = build_deepseek_v32_sp_context(
      attn_metadata, tokens, reinterpret_cast<ProcessGroup*>(0x1), 1, 4);
  ASSERT_TRUE(maybe_context.has_value());
  const auto& context = maybe_context.value();

  const int64_t padded_token_num =
      context.gathered_padded_reorder_index.size(0);
  torch::Tensor gathered = torch::zeros(
      {padded_token_num}, torch::TensorOptions().dtype(torch::kFloat32));
  auto* gathered_ptr = gathered.data_ptr<float>();
  auto pad_index = context.gathered_padded_reorder_index.to(torch::kCPU);
  const auto* pad_index_ptr = pad_index.data_ptr<int64_t>();
  for (int64_t i = 0; i < padded_token_num; ++i) {
    if (pad_index_ptr[i] == 10) {
      gathered_ptr[i] = -1.0f;
    } else {
      gathered_ptr[i] = static_cast<float>(pad_index_ptr[i]);
    }
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
  auto maybe_context = build_deepseek_v32_sp_context(
      attn_metadata, tokens, reinterpret_cast<ProcessGroup*>(0x1), 1, 4);
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
  auto maybe_context = build_deepseek_v32_sp_context(
      attn_metadata, tokens, reinterpret_cast<ProcessGroup*>(0x1), 1, 4);
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
