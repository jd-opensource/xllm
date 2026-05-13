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

#include "spec_input_builder.h"

#include <glog/logging.h>

#include <limits>

#include "framework/model/model_input_params.h"
#include "util/tensor_helper.h"

namespace xllm::specBuilder {

namespace {

template <typename T>
void pad_2d_vector(std::vector<std::vector<T>>& vec, T pad_value) {
  size_t max_col_size = 0;
  for (const std::vector<T>& row : vec) {
    max_col_size = std::max(max_col_size, row.size());
  }

  for (std::vector<T>& row : vec) {
    row.resize(max_col_size, pad_value);
  }
}

// Builds cumulative seq-lens layout: [0, l0, l0+l1, ...].
void push_cumsum(std::vector<int32_t>& vec, int32_t len) {
  if (vec.empty()) {
    vec.emplace_back(0);
  }
  vec.emplace_back(vec.back() + len);
}

// Resolves a row token from either input token_ids[seq_id] or row.token_id.
int32_t resolve_row_token_id(const DecodeCpuView& view, const RowSpec& row) {
  if (!row.use_input_token) {
    return row.token_id;
  }
  CHECK_LT(static_cast<size_t>(row.seq_id), view.token_ids.size())
      << "seq_id out of range for token_ids, seq_id=" << row.seq_id
      << ", token_ids_size=" << view.token_ids.size();
  return view.token_ids[row.seq_id];
}

Slice<int32_t> get_block_table_slice(const DecodeCpuView& view,
                                     int32_t seq_id) {
  CHECK_GE(seq_id, 0) << "invalid seq_id=" << seq_id;
  CHECK_LT(seq_id, view.num_sequences)
      << "seq_id out of range for block tables, seq_id=" << seq_id
      << ", num_sequences=" << view.num_sequences;
  CHECK_GT(view.block_table_row_stride, 0)
      << "invalid block table row stride=" << view.block_table_row_stride;

  const size_t row_offset =
      static_cast<size_t>(seq_id) * view.block_table_row_stride;
  CHECK_LE(row_offset + static_cast<size_t>(view.block_table_row_stride),
           view.block_tables_data.size())
      << "block table row out of range, seq_id=" << seq_id
      << ", row_offset=" << row_offset
      << ", row_stride=" << view.block_table_row_stride
      << ", block_tables_size=" << view.block_tables_data.size();
  return {view.block_tables_data.data() + row_offset,
          static_cast<size_t>(view.block_table_row_stride)};
}

}  // namespace

DecodeCpuView make_decode_cpu_view(const torch::Tensor& token_ids_cpu,
                                   const torch::Tensor& positions_cpu,
                                   const torch::Tensor& block_tables_cpu,
                                   const Slice<int32_t>& kv_seq_lens_slice) {
  DecodeCpuView view;
  if (token_ids_cpu.defined()) {
    view.token_ids = {token_ids_cpu.data_ptr<int32_t>(),
                      static_cast<size_t>(token_ids_cpu.numel())};
  }
  view.positions = {positions_cpu.data_ptr<int32_t>(),
                    static_cast<size_t>(positions_cpu.numel())};
  view.kv_seq_lens = kv_seq_lens_slice;
  CHECK(block_tables_cpu.defined()) << "block_tables_cpu is undefined";
  CHECK_EQ(block_tables_cpu.dim(), 2)
      << "block_tables_cpu must be 2D, got " << block_tables_cpu.sizes();
  view.block_tables_cpu = block_tables_cpu.contiguous();
  view.block_tables_data = {view.block_tables_cpu.data_ptr<int32_t>(),
                            static_cast<size_t>(view.block_tables_cpu.numel())};
  CHECK_LE(view.block_tables_cpu.size(0),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(view.block_tables_cpu.size(1),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  view.num_sequences = static_cast<int32_t>(view.block_tables_cpu.size(0));
  view.block_table_row_stride =
      static_cast<int32_t>(view.block_tables_cpu.size(1));
  return view;
}

int32_t calc_slot_id(int32_t position,
                     const Slice<int32_t>& block_table_slice,
                     int32_t block_size) {
  CHECK_GT(block_size, 0) << "invalid block_size=" << block_size;
  CHECK_GE(position, 0) << "invalid position=" << position;
  const int32_t block_idx = position / block_size;
  CHECK_LT(static_cast<size_t>(block_idx), block_table_slice.size())
      << "block table index out of range, block_idx=" << block_idx
      << ", block_table_size=" << block_table_slice.size()
      << ", position=" << position << ", block_size=" << block_size;
  const int32_t block_id = block_table_slice[block_idx];
  CHECK_GE(block_id, 0) << "invalid block_id=" << block_id;
  const int32_t block_offset = position % block_size;
  return block_id * block_size + block_offset;
}

int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset) {
  CHECK_GE(seq_id, 0) << "invalid seq_id=" << seq_id;
#if defined(USE_NPU)
  CHECK_LT(static_cast<size_t>(seq_id), kv_seq_lens_slice.size())
      << "seq_id out of range, seq_id=" << seq_id
      << ", kv_seq_lens_size=" << kv_seq_lens_slice.size();
  return kv_seq_lens_slice[seq_id] + offset;
#else
  CHECK_LT(static_cast<size_t>(seq_id + 1), kv_seq_lens_slice.size())
      << "seq_id out of range for cumulative layout, seq_id=" << seq_id
      << ", kv_seq_lens_size=" << kv_seq_lens_slice.size();
  return kv_seq_lens_slice[seq_id + 1] - kv_seq_lens_slice[seq_id] + offset;
#endif
}

void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len) {
#if defined(USE_NPU)
  vec.emplace_back(len);
#else
  push_cumsum(vec, len);
#endif
}

void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len) {
  if (kv_len > kv_max_seq_len) {
    kv_max_seq_len = kv_len;
  }
  append_seq_len_by_layout(kv_seq_lens_vec, kv_len);
}

void append_decode_row(const DecodeCpuView& view,
                       const RowSpec& row,
                       int32_t block_size,
                       DecodeBuildBuffers& buf) {
  CHECK_GE(row.seq_id, 0);
  CHECK_LT(row.seq_id, view.num_sequences);
  CHECK_LT(static_cast<size_t>(row.seq_id), view.positions.size());
  const int32_t new_position = view.positions[row.seq_id] + row.position_offset;
  CHECK_GE(new_position, 0) << "invalid decode position";

  const Slice<int32_t> block_table_slice =
      get_block_table_slice(view, row.seq_id);

  // All decode paths can toggle which fields are emitted, so one row builder
  // can serve draft/validate/first-decode/update-last-step scenarios.
  if (row.append_token) {
    buf.out_token_ids.emplace_back(resolve_row_token_id(view, row));
  }
  buf.out_positions.emplace_back(new_position);
  buf.out_new_cache_slots.emplace_back(
      calc_slot_id(new_position, block_table_slice, block_size));

  if (row.append_kv_len) {
    int32_t kv_len =
        calc_kv_len(view.kv_seq_lens, row.seq_id, row.position_offset);
    update_kv_seq_lens_and_max(buf.out_kv_seq_lens, kv_len, buf.kv_max_seq_len);
  }
  if (row.append_q_len_one) {
    append_seq_len_by_layout(buf.out_q_seq_lens, 1);
  }
  if (row.append_block_table) {
    buf.out_block_tables.emplace_back(block_table_slice.begin(),
                                      block_table_slice.end());
  }
}

TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    int32_t seq_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num) {
  CHECK_GT(last_step_decode_num, 0)
      << "invalid last_step_decode_num=" << last_step_decode_num;
  if (input_token_id >= 0) {
    TokenWithOffset direct;
    direct.token_id = input_token_id;
    direct.position_offset = 0;
    return direct;
  }

  const int32_t placeholder_idx = -input_token_id - 1;
  CHECK_GE(placeholder_idx, 0)
      << "invalid placeholder token id=" << input_token_id
      << ", seq_id=" << seq_id;
  const int32_t base_idx = placeholder_idx * last_step_decode_num;
  CHECK_LE(base_idx + last_step_decode_num,
           static_cast<int32_t>(last_step_tokens.size()))
      << "last_step_tokens out of range, seq_id=" << seq_id
      << ", placeholder_idx=" << placeholder_idx
      << ", last_step_decode_num=" << last_step_decode_num
      << ", last_step_tokens_size=" << last_step_tokens.size();

  TokenWithOffset resolved;
  resolved.position_offset = -1;
  for (int32_t i = 0; i < last_step_decode_num; ++i) {
    const int64_t candidate = last_step_tokens[base_idx + i];
    if (candidate >= 0) {
      CHECK_LE(candidate,
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
          << "token id overflow, seq_id=" << seq_id
          << ", candidate=" << candidate;
      resolved.token_id = static_cast<int32_t>(candidate);
      resolved.position_offset += 1;
    }
  }
  return resolved;
}

void append_decode_row_from_last_step(const DecodeCpuView& view,
                                      int32_t seq_id,
                                      int32_t input_token_id,
                                      const Slice<int64_t>& last_step_tokens,
                                      int32_t last_step_decode_num,
                                      int32_t block_size,
                                      DecodeBuildBuffers& buf) {
  // Placeholder tokens (-1/-2/...) are resolved from last-step outputs first,
  // then appended via the same row builder used by all decode paths.
  const TokenWithOffset resolved = resolve_token_with_position_offset(
      input_token_id, seq_id, last_step_tokens, last_step_decode_num);

  RowSpec row;
  row.seq_id = seq_id;
  row.token_id = resolved.token_id;
  row.position_offset = resolved.position_offset;
  append_decode_row(view, row, block_size, buf);
}

torch::Tensor build_q_cu_seq_lens_tensor(const ModelInputParams& params,
                                         torch::Device device) {
  std::vector<int32_t> q_cu_seq_lens_vec;
  q_cu_seq_lens_vec.reserve(params.num_sequences);
  int32_t cum_seq_len = 0;
  for (int32_t i = 0; i < params.num_sequences; ++i) {
    cum_seq_len += params.get_q_seq_len(i);
    q_cu_seq_lens_vec.emplace_back(cum_seq_len);
  }
  return torch::tensor(q_cu_seq_lens_vec,
                       torch::dtype(torch::kInt).device(device));
}

void update_input_params(ModelInputParams& input_params,
                         DecodeBuildBuffers& buf,
                         const torch::TensorOptions& int_options,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables,
                         torch::Device block_tables_device) {
  input_params.q_max_seq_len = q_max_seq_len;
  input_params.q_seq_lens_vec = std::move(q_seq_lens_vec);
  input_params.q_seq_lens =
      torch::tensor(input_params.q_seq_lens_vec, int_options);
  input_params.kv_max_seq_len = kv_max_seq_len;
  input_params.kv_seq_lens_vec = std::move(kv_seq_lens_vec);
  input_params.kv_seq_lens =
      torch::tensor(input_params.kv_seq_lens_vec, int_options);
  input_params.new_cache_slots =
      torch::tensor(buf.out_new_cache_slots, int_options);
  input_params.q_cu_seq_lens = build_q_cu_seq_lens_tensor(input_params);
  if (update_block_tables) {
    pad_2d_vector(buf.out_block_tables, /*pad_value=*/0);
    input_params.block_tables =
        create_2d_tensor(buf.out_block_tables, torch::kInt)
            .to(block_tables_device);
  }
}

namespace draftProbs {

namespace {

torch::Tensor extract_selected_probs(const torch::Tensor& draft_probs,
                                     const torch::Tensor& draft_token_ids) {
  CHECK(draft_probs.defined()) << "draft_probs must be defined";
  CHECK(draft_token_ids.defined()) << "draft_token_ids must be defined";

  if (draft_probs.dim() == 1) {
    return draft_probs;
  }

  if (draft_probs.dim() == 2) {
    CHECK_EQ(draft_probs.size(0), draft_token_ids.numel())
        << "draft_probs batch size mismatch";
    if (draft_probs.size(1) == 1) {
      return draft_probs.squeeze(-1);
    }
    auto ids = draft_token_ids.view({-1, 1}).to(torch::kLong);
    return draft_probs.gather(/*dim=*/-1, ids).squeeze(-1);
  }

  CHECK(false) << "draft_probs must be [batch], [batch,1] or [batch,vocab]";
  return torch::Tensor();
}

}  // namespace

torch::Tensor compress_for_cache(const torch::Tensor& draft_probs,
                                 const torch::Tensor& draft_token_ids) {
  return extract_selected_probs(draft_probs, draft_token_ids);
}

std::pair<torch::Tensor, torch::Tensor> build_validate_tensors(
    const std::vector<torch::Tensor>& draft_token_ids_steps,
    const std::vector<torch::Tensor>& draft_probs_steps,
    int32_t batch_size,
    int32_t vocab_size,
    bool enable_opt_validate_probs) {
  CHECK_GT(batch_size, 0) << "batch_size must be > 0";
  CHECK_GT(vocab_size, 0) << "vocab_size must be > 0";
  CHECK_EQ(draft_token_ids_steps.size(), draft_probs_steps.size())
      << "draft steps mismatch";
  CHECK(!draft_token_ids_steps.empty()) << "draft steps must not be empty";

  std::vector<torch::Tensor> token_ids_vec;
  std::vector<torch::Tensor> probs_vec;
  token_ids_vec.reserve(draft_token_ids_steps.size());
  probs_vec.reserve(draft_probs_steps.size());

  for (size_t i = 0; i < draft_token_ids_steps.size(); ++i) {
    auto draft_token_ids =
        draft_token_ids_steps[i].view({batch_size, 1}).to(torch::kLong);
    auto selected_probs =
        extract_selected_probs(draft_probs_steps[i], draft_token_ids)
            .view({batch_size, 1});

    token_ids_vec.push_back(draft_token_ids);
    if (enable_opt_validate_probs) {
      probs_vec.push_back(selected_probs);
    } else {
      auto dense_probs =
          torch::zeros({batch_size, 1, vocab_size}, selected_probs.options());
      dense_probs.scatter_(
          /*dim=*/-1,
          draft_token_ids.unsqueeze(-1),
          selected_probs.unsqueeze(-1));
      probs_vec.push_back(dense_probs);
    }
  }

  auto draft_token_ids = torch::cat(token_ids_vec, /*dim=*/1);
  auto draft_probs = torch::cat(probs_vec, /*dim=*/1);
  return {draft_token_ids, draft_probs};
}

}  // namespace draftProbs

}  // namespace xllm::specBuilder
