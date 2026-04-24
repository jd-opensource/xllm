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

#include "runtime/spec_input_builder.h"

#include <gtest/gtest.h>

#include <vector>

#include "framework/model/model_input_params.h"

namespace xllm {
namespace specBuilder {
namespace {

Slice<int32_t> to_slice(std::vector<int32_t>& vec) {
  return {vec.data(), static_cast<size_t>(vec.size())};
}

std::vector<int32_t> to_layout_seq_lens(const std::vector<int32_t>& lens) {
#if defined(USE_NPU)
  return lens;
#else
  std::vector<int32_t> out;
  out.reserve(lens.size() + 1);
  out.emplace_back(0);
  int32_t sum = 0;
  for (int32_t len : lens) {
    sum += len;
    out.emplace_back(sum);
  }
  return out;
#endif
}

std::vector<int32_t> tensor_to_vec_int32(const torch::Tensor& tensor) {
  torch::Tensor cpu_tensor =
      tensor.to(torch::kCPU).to(torch::kInt).contiguous();
  const int32_t* data = cpu_tensor.data_ptr<int32_t>();
  return {data, data + cpu_tensor.numel()};
}

TEST(SpecDecodeInputBuilderTest, DraftInputsSingleRowPerSeq) {
  ModelInputParams params;
  params.num_sequences = 2;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});

  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  auto view = make_decode_cpu_view(
      torch::Tensor(), positions, block_tables, to_slice(kv_seq_lens));

  DecodeBuildBuffers buf;
  for (int32_t seq_id = 0; seq_id < params.num_sequences; ++seq_id) {
    RowSpec row;
    row.seq_id = seq_id;
    row.position_offset = 1;
    row.append_token = false;
    append_decode_row(view, row, /*block_size=*/4, buf);
  }

  EXPECT_TRUE(buf.out_token_ids.empty());
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 9}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({5, 21}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 10}));
}

TEST(SpecDecodeInputBuilderTest, ValidateInputsNonAtbExpansion) {
  ModelInputParams params;
  params.num_sequences = 2;
  const int32_t num_speculative_tokens = 2;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});

  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  auto view = make_decode_cpu_view(
      token_ids, positions, block_tables, to_slice(kv_seq_lens));

  DecodeBuildBuffers buf;
  for (int32_t seq_id = 0; seq_id < params.num_sequences; ++seq_id) {
    for (int32_t val_idx = 0; val_idx < num_val_tokens; ++val_idx) {
      RowSpec row;
      row.seq_id = seq_id;
      if (val_idx == 0) {
        row.use_input_token = true;
      } else {
        row.token_id = -1 * val_idx;
      }
      row.position_offset = 1 + val_idx;
      row.append_q_len_one = true;
      row.append_block_table = true;
      append_decode_row(view, row, /*block_size=*/4, buf);
    }
  }

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({10, -1, -2, 20, -1, -2}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 6, 7, 9, 10, 11}));
  EXPECT_EQ(buf.out_new_cache_slots,
            std::vector<int32_t>({5, 6, 7, 21, 22, 23}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 7, 8, 10, 11, 12}));
  EXPECT_EQ(buf.out_q_seq_lens, to_layout_seq_lens({1, 1, 1, 1, 1, 1}));
  ASSERT_EQ(buf.out_block_tables.size(), 6);
}

TEST(SpecDecodeInputBuilderTest, AppendDecodeRowTokenKinds) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  auto view = make_decode_cpu_view(
      token_ids, positions, block_tables, to_slice(kv_seq_lens));

  DecodeBuildBuffers buf;
  append_decode_row(
      view,
      {.seq_id = 1, .use_input_token = true, .position_offset = 0},
      /*block_size=*/4,
      buf);
  append_decode_row(view,
                    {.seq_id = 0, .token_id = 123, .position_offset = 0},
                    /*block_size=*/4,
                    buf);
  append_decode_row(view,
                    {.seq_id = 0, .token_id = -2, .position_offset = 0},
                    /*block_size=*/4,
                    buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({20, 123, -2}));
}

TEST(SpecDecodeInputBuilderTest, MakeDecodeCpuViewUsesFlatBlockTableLayout) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2, 0}, {3, 4, 5, 0}}, torch::kInt);

  DecodeCpuView view = make_decode_cpu_view(
      token_ids, positions, block_tables, to_slice(kv_seq_lens));

  EXPECT_EQ(view.num_sequences, 2);
  EXPECT_EQ(view.block_table_row_stride, 4);
  EXPECT_EQ(view.block_tables_data,
            std::vector<int32_t>({0, 1, 2, 0, 3, 4, 5, 0}));

  ModelInputParams params;
  params.num_sequences = 2;
  DecodeBuildBuffers buf;
  append_decode_row(view,
                    {.seq_id = 1, .token_id = 99, .position_offset = 2},
                    /*block_size=*/4,
                    buf);

  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({10}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({22}));
  ASSERT_EQ(buf.out_block_tables.size(), 0);
}

TEST(SpecDecodeInputBuilderTest, ValidateRowsStartFromCorrectedCurrentView) {
  ModelInputParams params;
  params.num_sequences = 2;
  std::vector<int32_t> token_ids = {31, 41};
  std::vector<int32_t> positions = {6, 9};
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({7, 10});

  DecodeCpuView view;
  view.token_ids = token_ids;
  view.positions = positions;
  view.kv_seq_lens = kv_seq_lens;
  view.block_tables_cpu =
      torch::tensor({{0, 1, 2, 0}, {3, 4, 5, 0}}, torch::kInt);
  view.block_tables_data = {view.block_tables_cpu.data_ptr<int32_t>(),
                            static_cast<size_t>(view.block_tables_cpu.numel())};
  view.num_sequences = static_cast<int32_t>(view.block_tables_cpu.size(0));
  view.block_table_row_stride =
      static_cast<int32_t>(view.block_tables_cpu.size(1));

  DecodeBuildBuffers buf;
  append_decode_row(
      view,
      {.seq_id = 0, .token_id = view.token_ids[0], .position_offset = 0},
      /*block_size=*/4,
      buf);
  append_decode_row(view,
                    {.seq_id = 0, .token_id = -1, .position_offset = 1},
                    /*block_size=*/4,
                    buf);
  append_decode_row(
      view,
      {.seq_id = 1, .token_id = view.token_ids[1], .position_offset = 0},
      /*block_size=*/4,
      buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({31, -1, 41}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({6, 7, 9}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({6, 7, 21}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({7, 8, 10}));
}

TEST(SpecDecodeInputBuilderTest, ValidateInputsAtbChunkedPrefillShape) {
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});
  std::vector<int32_t> atb_kv_seq_lens;
  std::vector<int32_t> atb_q_seq_lens;
  int32_t atb_kv_max_seq_len = 0;
  const int32_t num_val_tokens = 3;

  auto kv_slice = to_slice(kv_seq_lens);
  for (int32_t seq_id = 0; seq_id < 2; ++seq_id) {
    int32_t kv_len = calc_kv_len(kv_slice, seq_id, /*offset=*/0);
    int32_t kv_len_after_validation = kv_len + num_val_tokens;
    update_kv_seq_lens_and_max(
        atb_kv_seq_lens, kv_len_after_validation, atb_kv_max_seq_len);
    append_seq_len_by_layout(atb_q_seq_lens, num_val_tokens);
  }

  EXPECT_EQ(atb_kv_seq_lens, to_layout_seq_lens({8, 12}));
  EXPECT_EQ(atb_q_seq_lens, to_layout_seq_lens({3, 3}));
  EXPECT_EQ(atb_kv_max_seq_len, 12);
}

TEST(SpecDecodeInputBuilderTest, FirstDecodeInputsFixAndNonFixMix) {
  ModelInputParams params;
  params.num_sequences = 2;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({6, 9});

  torch::Tensor token_ids = torch::tensor({100, 200}, torch::kInt);
  torch::Tensor positions = torch::tensor({5, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  auto view = make_decode_cpu_view(
      token_ids, positions, block_tables, to_slice(kv_seq_lens));

  DecodeBuildBuffers buf;
  std::vector<int32_t> select_row_idx(2, 0);
  auto emit_row =
      [&](int32_t seq_id, int32_t token_id, int32_t position_offset) {
        RowSpec row;
        row.seq_id = seq_id;
        row.token_id = token_id;
        row.position_offset = position_offset;
        row.append_q_len_one = true;
        row.append_block_table = true;
        append_decode_row(view, row, /*block_size=*/4, buf);
      };

  emit_row(/*seq_id=*/0, /*token_id=*/90, /*position_offset=*/-1);
  emit_row(/*seq_id=*/0, /*token_id=*/100, /*position_offset=*/0);
  select_row_idx[0] = static_cast<int32_t>(buf.out_token_ids.size()) - 1;

  emit_row(/*seq_id=*/1, /*token_id=*/200, /*position_offset=*/0);
  select_row_idx[1] = static_cast<int32_t>(buf.out_token_ids.size()) - 1;

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({90, 100, 200}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({4, 5, 8}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({4, 5, 20}));
  EXPECT_EQ(buf.out_q_seq_lens, to_layout_seq_lens({1, 1, 1}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({5, 6, 9}));
  EXPECT_EQ(select_row_idx, std::vector<int32_t>({1, 2}));
  ASSERT_EQ(buf.out_block_tables.size(), 3);
}

TEST(SpecDecodeInputBuilderTest, AppendDecodeRowWithInputTokenSource) {
  ModelInputParams params;
  params.num_sequences = 2;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({5, 9});

  torch::Tensor token_ids = torch::tensor({10, 20}, torch::kInt);
  torch::Tensor positions = torch::tensor({4, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  auto view = make_decode_cpu_view(
      token_ids, positions, block_tables, to_slice(kv_seq_lens));

  DecodeBuildBuffers buf;
  append_decode_row(view,
                    {.seq_id = 0,
                     .use_input_token = true,
                     .position_offset = 1,
                     .append_q_len_one = true,
                     .append_block_table = true},
                    /*block_size=*/4,
                    buf);
  append_decode_row(view,
                    {.seq_id = 1,
                     .token_id = -2,
                     .position_offset = 2,
                     .append_q_len_one = true,
                     .append_block_table = true},
                    /*block_size=*/4,
                    buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({10, -2}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 10}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({5, 22}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 11}));
  EXPECT_EQ(buf.out_q_seq_lens, to_layout_seq_lens({1, 1}));
  ASSERT_EQ(buf.out_block_tables.size(), 2);
}

TEST(SpecDecodeInputBuilderTest, ResolveTokenWithPositionOffset) {
  std::vector<int64_t> last_step_tokens = {11, -1, 13, -1, -1, -1};
  Slice<int64_t> last_step_slice = {
      last_step_tokens.data(), static_cast<size_t>(last_step_tokens.size())};

  TokenWithOffset direct =
      resolve_token_with_position_offset(/*input_token_id=*/20,
                                         /*seq_id=*/0,
                                         last_step_slice,
                                         /*last_step_decode_num=*/3);
  EXPECT_EQ(direct.token_id, 20);
  EXPECT_EQ(direct.position_offset, 0);

  TokenWithOffset resolved =
      resolve_token_with_position_offset(/*input_token_id=*/-1,
                                         /*seq_id=*/0,
                                         last_step_slice,
                                         /*last_step_decode_num=*/3);
  EXPECT_EQ(resolved.token_id, 13);
  EXPECT_EQ(resolved.position_offset, 1);

  TokenWithOffset no_accept =
      resolve_token_with_position_offset(/*input_token_id=*/-2,
                                         /*seq_id=*/1,
                                         last_step_slice,
                                         /*last_step_decode_num=*/3);
  EXPECT_EQ(no_accept.token_id, 0);
  EXPECT_EQ(no_accept.position_offset, -1);
}

TEST(SpecDecodeInputBuilderTest, AppendDecodeRowFromLastStep) {
  ModelInputParams params;
  params.num_sequences = 2;
  std::vector<int32_t> kv_seq_lens = to_layout_seq_lens({6, 9});

  torch::Tensor token_ids = torch::tensor({100, -1}, torch::kInt);
  torch::Tensor positions = torch::tensor({5, 8}, torch::kInt);
  torch::Tensor block_tables =
      torch::tensor({{0, 1, 2}, {3, 4, 5}}, torch::kInt);
  auto view = make_decode_cpu_view(
      token_ids, positions, block_tables, to_slice(kv_seq_lens));

  std::vector<int64_t> last_step_tokens = {201, 202};
  Slice<int64_t> last_step_slice = {
      last_step_tokens.data(), static_cast<size_t>(last_step_tokens.size())};

  DecodeBuildBuffers buf;
  append_decode_row_from_last_step(view,
                                   /*seq_id=*/0,
                                   /*input_token_id=*/view.token_ids[0],
                                   last_step_slice,
                                   /*last_step_decode_num=*/2,
                                   /*block_size=*/4,
                                   buf);
  append_decode_row_from_last_step(view,
                                   /*seq_id=*/1,
                                   /*input_token_id=*/view.token_ids[1],
                                   last_step_slice,
                                   /*last_step_decode_num=*/2,
                                   /*block_size=*/4,
                                   buf);

  EXPECT_EQ(buf.out_token_ids, std::vector<int32_t>({100, 202}));
  EXPECT_EQ(buf.out_positions, std::vector<int32_t>({5, 9}));
  EXPECT_EQ(buf.out_new_cache_slots, std::vector<int32_t>({5, 21}));
  EXPECT_EQ(buf.out_kv_seq_lens, to_layout_seq_lens({6, 10}));
}

TEST(SpecDecodeInputBuilderTest, QCuSeqLensConsistency) {
  ModelInputParams params;
  params.num_sequences = 3;
  params.q_seq_lens_vec = to_layout_seq_lens({1, 2, 3});

  torch::Tensor q_cu_seq_lens = build_q_cu_seq_lens_tensor(params);
  EXPECT_EQ(tensor_to_vec_int32(q_cu_seq_lens),
            std::vector<int32_t>({1, 3, 6}));
}

TEST(SpecDecodeInputBuilderTest, CalcSlotIdOutOfRangeDeath) {
  std::vector<int32_t> block_table = {0};
  EXPECT_DEATH(calc_slot_id(/*position=*/4,
                            to_slice(block_table),
                            /*block_size=*/4),
               "block table index out of range");
}

TEST(DraftProbsBuilderTest, CompressForCacheDense) {
  auto draft_probs =
      torch::tensor({{0.1f, 0.2f, 0.7f}, {0.6f, 0.1f, 0.3f}}, torch::kFloat32);
  auto token_ids = torch::tensor({1, 0}, torch::kInt64);
  auto compressed = draftProbs::compress_for_cache(draft_probs, token_ids);
  auto expected = torch::tensor({0.2f, 0.6f}, torch::kFloat32);
  EXPECT_TRUE(torch::allclose(compressed, expected));
}

TEST(DraftProbsBuilderTest, BuildValidateTensorsSelectedOnly) {
  std::vector<torch::Tensor> token_steps = {
      torch::tensor({3, 4}, torch::kInt64),
      torch::tensor({5, 6}, torch::kInt64)};
  std::vector<torch::Tensor> probs_steps = {
      torch::tensor({0.3f, 0.4f}, torch::kFloat32),
      torch::tensor({0.5f, 0.6f}, torch::kFloat32)};

  auto [draft_token_ids, draft_probs] =
      draftProbs::build_validate_tensors(token_steps,
                                         probs_steps,
                                         /*batch_size=*/2,
                                         /*vocab_size=*/8,
                                         /*enable_opt_validate_probs=*/true);

  EXPECT_EQ(draft_token_ids.dim(), 2);
  EXPECT_EQ(draft_probs.dim(), 2);
  EXPECT_EQ(draft_token_ids.size(0), 2);
  EXPECT_EQ(draft_token_ids.size(1), 2);
  EXPECT_EQ(draft_probs.size(0), 2);
  EXPECT_EQ(draft_probs.size(1), 2);
  EXPECT_TRUE(torch::allclose(
      draft_probs,
      torch::tensor({{0.3f, 0.5f}, {0.4f, 0.6f}}, torch::kFloat32)));
}

TEST(DraftProbsBuilderTest, BuildValidateTensorsRecoveredDense) {
  std::vector<torch::Tensor> token_steps = {
      torch::tensor({1, 2}, torch::kInt64),
      torch::tensor({0, 3}, torch::kInt64)};
  std::vector<torch::Tensor> probs_steps = {
      torch::tensor({0.2f, 0.7f}, torch::kFloat32),
      torch::tensor({0.9f, 0.1f}, torch::kFloat32)};

  auto [draft_token_ids, draft_probs] =
      draftProbs::build_validate_tensors(token_steps,
                                         probs_steps,
                                         /*batch_size=*/2,
                                         /*vocab_size=*/5,
                                         /*enable_opt_validate_probs=*/false);

  EXPECT_EQ(draft_token_ids.dim(), 2);
  EXPECT_EQ(draft_probs.dim(), 3);
  EXPECT_EQ(draft_probs.size(0), 2);
  EXPECT_EQ(draft_probs.size(1), 2);
  EXPECT_EQ(draft_probs.size(2), 5);

  auto selected =
      draft_probs.gather(/*dim=*/-1, draft_token_ids.unsqueeze(-1)).squeeze(-1);
  auto expected_selected =
      torch::tensor({{0.2f, 0.9f}, {0.7f, 0.1f}}, torch::kFloat32);
  EXPECT_TRUE(torch::allclose(selected, expected_selected));

  auto row_sums = draft_probs.sum(/*dim=*/-1);
  EXPECT_TRUE(torch::allclose(row_sums, expected_selected));
}

TEST(DraftProbsBuilderTest, BuildValidateTensorsDenseInputFallback) {
  std::vector<torch::Tensor> token_steps = {
      torch::tensor({2, 1}, torch::kInt64)};
  std::vector<torch::Tensor> probs_steps = {
      torch::tensor({{0.1f, 0.2f, 0.7f}, {0.3f, 0.6f, 0.1f}}, torch::kFloat32)};

  auto [draft_token_ids, draft_probs] =
      draftProbs::build_validate_tensors(token_steps,
                                         probs_steps,
                                         /*batch_size=*/2,
                                         /*vocab_size=*/3,
                                         /*enable_opt_validate_probs=*/true);

  EXPECT_EQ(draft_token_ids.dim(), 2);
  EXPECT_EQ(draft_token_ids.size(0), 2);
  EXPECT_EQ(draft_token_ids.size(1), 1);
  EXPECT_EQ(draft_probs.dim(), 2);
  EXPECT_EQ(draft_probs.size(0), 2);
  EXPECT_EQ(draft_probs.size(1), 1);
  EXPECT_TRUE(torch::allclose(
      draft_probs, torch::tensor({{0.7f}, {0.6f}}, torch::kFloat32)));
}

}  // namespace
}  // namespace specBuilder
}  // namespace xllm
