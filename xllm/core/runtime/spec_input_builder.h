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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "util/slice.h"

namespace xllm {

struct ModelInputParams;

namespace specBuilder {

struct DecodeCpuView {
  torch::Tensor token_ids_cpu;
  torch::Tensor positions_cpu;
  torch::Tensor block_tables_cpu;
  std::vector<torch::Tensor> multi_block_tables_cpu;
  std::vector<int32_t> token_ids_vec;
  std::vector<int32_t> positions_vec;
  std::vector<int32_t> kv_seq_lens_vec;
  Slice<int32_t> token_ids;
  Slice<int32_t> positions;
  Slice<int32_t> kv_seq_lens;
  std::vector<Slice<int32_t>> block_table_slices;
  std::vector<std::vector<Slice<int32_t>>> multi_block_table_slices;
  bool model_managed_multiblock = false;
};

struct DecodeBuildBuffers {
  std::vector<int32_t> out_token_ids;
  std::vector<int32_t> out_positions;
  std::vector<int32_t> out_kv_seq_lens;
  std::vector<int32_t> out_q_seq_lens;
  std::vector<int32_t> out_new_cache_slots;
  std::vector<std::vector<int32_t>> out_block_tables;
  std::vector<std::vector<std::vector<int32_t>>> out_multi_block_tables;
  int32_t kv_max_seq_len = 0;
};

struct RowSpec {
  int32_t seq_id = 0;
  bool use_input_token = false;
  int32_t token_id = 0;
  int32_t position_offset = 0;
  bool append_token = true;
  bool append_kv_len = true;
  bool append_q_len_one = false;
  bool append_block_table = false;
};

struct TokenWithOffset {
  int32_t token_id = 0;
  int32_t position_offset = 0;
};

DecodeCpuView make_decode_cpu_view(const torch::Tensor& token_ids_cpu,
                                   const torch::Tensor& positions_cpu,
                                   const ModelInputParams& params);

void append_decode_row(const DecodeCpuView& view,
                       const RowSpec& row,
                       int32_t block_size,
                       DecodeBuildBuffers& buf);

TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    int32_t seq_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num);

void append_decode_row_from_last_step(const DecodeCpuView& view,
                                      int32_t seq_id,
                                      int32_t input_token_id,
                                      const Slice<int64_t>& last_step_tokens,
                                      int32_t last_step_decode_num,
                                      int32_t block_size,
                                      DecodeBuildBuffers& buf);

int32_t calc_slot_id(int32_t position,
                     const Slice<int32_t>& block_table_slice,
                     int32_t block_size);

int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset);

void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len);

void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len);

torch::Tensor build_q_cu_seq_lens_tensor(const ModelInputParams& params,
                                         torch::Device device = torch::kCPU);

void update_input_params(ModelInputParams& input_params,
                         DecodeBuildBuffers& buf,
                         const torch::TensorOptions& int_options,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables = false,
                         torch::Device block_tables_device = torch::kCPU);

namespace draftProbs {

torch::Tensor compress_for_cache(const torch::Tensor& draft_probs,
                                 const torch::Tensor& draft_token_ids);

std::pair<torch::Tensor, torch::Tensor> build_validate_tensors(
    const std::vector<torch::Tensor>& draft_token_ids_steps,
    const std::vector<torch::Tensor>& draft_probs_steps,
    int32_t batch_size,
    int32_t vocab_size,
    bool enable_opt_validate_probs);

}  // namespace draftProbs

}  // namespace specBuilder

}  // namespace xllm
